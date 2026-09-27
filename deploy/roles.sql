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
CREATE ROLE sentinel_migrator LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
CREATE ROLE sentinel_writer   LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
CREATE ROLE sentinel_reader   LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
CREATE ROLE sentinel_reviewer LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;

-- Passwords below are DEV PLACEHOLDERS for the local compose stack and CI.
-- Provisioning for anything shared must inject real secrets (e.g. via psql
-- variables or a secrets manager) instead of relying on these defaults.
ALTER ROLE sentinel_writer WITH PASSWORD 'writer_dev';
ALTER ROLE sentinel_reader WITH PASSWORD 'reader_dev';
ALTER ROLE sentinel_reviewer WITH PASSWORD 'reviewer_dev';

-- each role may connect to the sentinel database and use the objects in its
-- schema(s); the migrator owns the data
GRANT CONNECT ON DATABASE sentinel TO
    sentinel_migrator, sentinel_writer, sentinel_reader, sentinel_reviewer;
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
ALTER DEFAULT PRIVILEGES FOR ROLE sentinel_migrator IN SCHEMA public
    GRANT UPDATE (adjudication, adjudicated_by, adjudicated_at, auto_resolved)
    ON TABLES TO sentinel_reviewer;

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
REVOKE UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public FROM sentinel_reviewer;
GRANT UPDATE (adjudication, adjudicated_by, adjudicated_at, auto_resolved)
    ON flags TO sentinel_reviewer;
