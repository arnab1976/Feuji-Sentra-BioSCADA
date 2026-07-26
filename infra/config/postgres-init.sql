-- BioSCADA AI — audit schema.
-- Append-only with hash chaining gives tamper-evidence without a
-- commercial WORM appliance (21 CFR Part 11 supporting control).

CREATE TABLE IF NOT EXISTS audit_trail (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    level         TEXT        NOT NULL,
    action        TEXT        NOT NULL,
    event_id      TEXT,
    actor         TEXT,
    detail        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    prev_hash     TEXT,
    row_hash      TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_trail(event_id);
CREATE INDEX IF NOT EXISTS idx_audit_ts    ON audit_trail(ts DESC);

-- hash-chain trigger: each row commits to its predecessor
CREATE OR REPLACE FUNCTION audit_hash_chain() RETURNS TRIGGER AS $$
DECLARE last_hash TEXT;
BEGIN
    SELECT row_hash INTO last_hash FROM audit_trail ORDER BY id DESC LIMIT 1;
    NEW.prev_hash := COALESCE(last_hash, 'GENESIS');
    NEW.row_hash  := encode(sha256((NEW.prev_hash || NEW.ts::text || NEW.action
                       || COALESCE(NEW.event_id,'') || NEW.detail::text)::bytea), 'hex');
    RETURN NEW;
END; $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_hash ON audit_trail;
CREATE TRIGGER trg_audit_hash BEFORE INSERT ON audit_trail
    FOR EACH ROW EXECUTE FUNCTION audit_hash_chain();

-- block UPDATE/DELETE: audit rows are immutable
CREATE OR REPLACE FUNCTION audit_immutable() RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'audit_trail is append-only (21 CFR Part 11)';
END; $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_no_update ON audit_trail;
CREATE TRIGGER trg_audit_no_update BEFORE UPDATE OR DELETE ON audit_trail
    FOR EACH ROW EXECUTE FUNCTION audit_immutable();

CREATE TABLE IF NOT EXISTS signatures (
    event_id   TEXT PRIMARY KEY,
    signer     TEXT NOT NULL,
    role       TEXT NOT NULL,
    reason     TEXT NOT NULL,
    signed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    committed  BOOLEAN NOT NULL
);
