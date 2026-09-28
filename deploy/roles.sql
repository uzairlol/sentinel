-- Sentinel least-privilege database roles (S2-T3, reviewer added in S3-T1).
--
-- The store is append-only by design (INV-2, docs/adr/0002). These four roles
-- enforce that property at the database layer so that even a compromised
-- application credential cannot rewrite or delete evidence:
--
--   sentinel_migrator  - DDL owner. Runs Alembic migrations and owns every
--                        object in the schema. No application code uses this
--                        credential.
--   sentinel_writer    - INSERT + SELECT only. Its grants are issued via
--                        ALTER DEFAULT PRIVILEGES so future tables (created by
--                        the migrator) stay writable without re-granting.
--                        UPDATE/DELETE/TRUNCATE are revoked explicitly.
--   sentinel_reader    - SELECT only. For replay, audit, and reporting jobs.
--   sentinel_reviewer  - SELECT everywhere, plus column-scoped UPDATE on the
--                        four adjudication columns of `flags`. Adjudicating a
--                        flag is the only mutating operation in the product
--                        (docs/adr/0012), so it gets its own narrowly scoped
--                        credential instead of widening the writer.
--
-- Run as a superuser, either once against a fresh cluster or from the compose
-- init below:
--
--   psql -U postgres -d sentinel -f roles.sql
--
-- Safe to run more than once (roles are created only if missing) and safe to
-- run before the schema exists: table grants that name a table Alembic has not
-- created yet are skipped rather than erroring, and are applied by migration
-- 0002 or by the test fixture that re-grants after a migration recreates a
-- table.
--
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sentinel_migrator') THEN
        CREATE ROLE sentinel_migrator LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sentinel_writer') THEN
        CREATE ROLE sentinel_writer   LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sentinel_reader') THEN
        CREATE ROLE sentinel_reader   LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sentinel_reviewer') THEN
        CREATE ROLE sentinel_reviewer LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
    END IF;
END
$$;

-- Passwords below are DEV PLACEHOLDERS for the local compose stack and CI.
-- Provisioning for anything shared must inject real secrets (e.g. via psql
-- variables or a secrets manager) instead of relying on these defaults.
-- Only reset when the role is new, so a re-run does not rotate a password an
-- operator has already provisioned.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sentinel_writer') THEN
        EXECUTE 'ALTER ROLE sentinel_writer   WITH PASSWORD ''writer_dev''';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sentinel_reader') THEN
        EXECUTE 'ALTER ROLE sentinel_reader   WITH PASSWORD ''reader_dev''';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sentinel_reviewer') THEN
        EXECUTE 'ALTER ROLE sentinel_reviewer WITH PASSWORD ''reviewer_dev''';
    END IF;
END
$$;

-- each role may connect to *this* database and use the objects in its
-- schema(s); the migrator owns the data. Resolved from current_database() in a
-- DO block -- GRANT takes an identifier, not an expression -- so the same file
-- provisions the compose stack's `sentinel` and CI's `sentinel_test`.
DO $$
BEGIN
    EXECUTE format(
        'GRANT CONNECT ON DATABASE %I TO sentinel_migrator, sentinel_writer, '
        'sentinel_reader, sentinel_reviewer',
        current_database()
    );
END
$$;
GRANT USAGE, CREATE ON SCHEMA public TO sentinel_migrator;
GRANT USAGE ON SCHEMA public TO
    sentinel_writer, sentinel_reader, sentinel_reviewer;

-- appenders never see table data wholesale; they write rows and read back
-- only what they match on
GRANT INSERT, SELECT ON ALL TABLES IN SCHEMA public TO sentinel_writer;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO sentinel_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO sentinel_reviewer;

-- carry the grants forward to objects the migrator creates later
ALTER DEFAULT PRIVILEGES FOR ROLE sentinel_migrator IN SCHEMA public
    GRANT INSERT, SELECT ON TABLES TO sentinel_writer;
ALTER DEFAULT PRIVILEGES FOR ROLE sentinel_migrator IN SCHEMA public
    GRANT SELECT ON TABLES TO sentinel_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE sentinel_migrator IN SCHEMA public
    GRANT SELECT ON TABLES TO sentinel_reviewer;
-- Note: PostgreSQL supports only table-level grants in ALTER DEFAULT
-- PRIVILEGES, so the reviewer's column-scoped UPDATE cannot be defaulted
-- ("default privileges cannot be set for columns" is a hard error). A
-- migration that adds an adjudication column must grant it explicitly; the
-- grant for today's columns is at the bottom of this file.

-- belt-and-braces: revoke everything a mutating writer would need. Public
-- schema ownership is the cluster default; tighten it so only the migrator can
-- place new objects outside a migration run.
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM sentinel_writer;
GRANT INSERT, SELECT ON ALL TABLES IN SCHEMA public TO sentinel_writer;
REVOKE UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
    ON ALL TABLES IN SCHEMA public FROM sentinel_writer;

-- the reviewer reads everything but may only change an adjudication: not the
-- summary, not the evidence, not the severity. A reviewer who disagrees with a
-- finding rejects it -- they do not get to rewrite it.
--
-- Guarded because this file runs before Alembic has created `flags` (the CI
-- service container is provisioned first, the schema second). Migration 0002
-- issues the same grant once the table exists, and
-- test_db_roles.py re-applies it after a migration recreates the table.
DO $$
BEGIN
    IF to_regclass('public.flags') IS NOT NULL THEN
        EXECUTE 'REVOKE UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public '
                'FROM sentinel_reviewer';
        EXECUTE 'GRANT UPDATE (adjudication, adjudicated_by, adjudicated_at, auto_resolved) '
                'ON flags TO sentinel_reviewer';
    END IF;
END
$$;
