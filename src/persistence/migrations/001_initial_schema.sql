-- NeuralSleep Initial Database Schema
-- Migration: 001_initial_schema.sql
-- Created: 2025-12-29

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =====================================================
-- Users Table
-- =====================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMPTZ,
    settings JSONB DEFAULT '{}'::JSONB,
    metadata JSONB DEFAULT '{}'::JSONB
);

CREATE INDEX IF NOT EXISTS idx_users_external_id ON users(external_id);
CREATE INDEX IF NOT EXISTS idx_users_last_active ON users(last_active_at);
CREATE INDEX IF NOT EXISTS idx_users_created ON users(created_at);

-- =====================================================
-- Semantic Memory States
-- Stores 256-dimensional tensors as BYTEA
-- =====================================================
CREATE TABLE IF NOT EXISTS semantic_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    state_tensor BYTEA NOT NULL,
    tensor_size INTEGER NOT NULL DEFAULT 256,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::JSONB,
    CONSTRAINT unique_user_semantic UNIQUE (user_id)
);

CREATE INDEX IF NOT EXISTS idx_semantic_states_user ON semantic_states(user_id);
CREATE INDEX IF NOT EXISTS idx_semantic_states_updated ON semantic_states(updated_at);

-- =====================================================
-- Episodic Experiences
-- Stores 128-dimensional tensors with importance and timestamps
-- =====================================================
CREATE TABLE IF NOT EXISTS episodic_experiences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    experience_tensor BYTEA NOT NULL,
    tensor_size INTEGER NOT NULL DEFAULT 128,
    importance FLOAT NOT NULL DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    event_timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50),
    character_id VARCHAR(50),
    correct BOOLEAN,
    time_spent_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    consolidated BOOLEAN DEFAULT FALSE,
    consolidated_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::JSONB
);

CREATE INDEX IF NOT EXISTS idx_episodic_user ON episodic_experiences(user_id);
CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic_experiences(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_episodic_importance ON episodic_experiences(importance DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_unconsolidated ON episodic_experiences(user_id, consolidated)
    WHERE NOT consolidated;
CREATE INDEX IF NOT EXISTS idx_episodic_user_time ON episodic_experiences(user_id, event_timestamp DESC);

-- =====================================================
-- Working Memory Buffers
-- Transient experiences with TTL for cleanup
-- =====================================================
CREATE TABLE IF NOT EXISTS working_memory_buffers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    state_tensor BYTEA NOT NULL,
    output_tensor BYTEA NOT NULL,
    importance FLOAT NOT NULL DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    event_timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}'::JSONB
);

CREATE INDEX IF NOT EXISTS idx_working_buffer_user ON working_memory_buffers(user_id);
CREATE INDEX IF NOT EXISTS idx_working_buffer_timestamp ON working_memory_buffers(event_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_working_buffer_expires ON working_memory_buffers(expires_at);
CREATE INDEX IF NOT EXISTS idx_working_buffer_user_time ON working_memory_buffers(user_id, event_timestamp DESC);

-- =====================================================
-- Working Memory Current States
-- Per-user current hidden state
-- =====================================================
CREATE TABLE IF NOT EXISTS working_memory_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    state_tensor BYTEA NOT NULL,
    tensor_size INTEGER NOT NULL DEFAULT 256,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::JSONB,
    CONSTRAINT unique_user_working_state UNIQUE (user_id)
);

CREATE INDEX IF NOT EXISTS idx_working_states_user ON working_memory_states(user_id);

-- =====================================================
-- Consciousness Metrics History
-- =====================================================
CREATE TABLE IF NOT EXISTS consciousness_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    integrated_information FLOAT CHECK (integrated_information >= 0 AND integrated_information <= 1),
    self_reference_depth INTEGER CHECK (self_reference_depth >= 0),
    temporal_integration FLOAT CHECK (temporal_integration >= 0 AND temporal_integration <= 1),
    causal_density FLOAT CHECK (causal_density >= 0 AND causal_density <= 1),
    dynamical_complexity FLOAT CHECK (dynamical_complexity >= 0 AND dynamical_complexity <= 1),
    information_flow JSONB,
    consciousness_level VARCHAR(50),
    working_state_hash VARCHAR(64),
    episodic_state_hash VARCHAR(64),
    semantic_state_hash VARCHAR(64),
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    computation_time_ms INTEGER,
    metadata JSONB DEFAULT '{}'::JSONB
);

CREATE INDEX IF NOT EXISTS idx_consciousness_computed ON consciousness_metrics(computed_at DESC);
CREATE INDEX IF NOT EXISTS idx_consciousness_phi ON consciousness_metrics(integrated_information);
CREATE INDEX IF NOT EXISTS idx_consciousness_level ON consciousness_metrics(consciousness_level);

-- =====================================================
-- Consciousness State History
-- Combined state snapshots for Phi computation
-- =====================================================
CREATE TABLE IF NOT EXISTS consciousness_state_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    combined_state BYTEA NOT NULL,
    state_dimensions INTEGER NOT NULL,
    working_contribution FLOAT,
    episodic_contribution FLOAT,
    semantic_contribution FLOAT,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::JSONB
);

CREATE INDEX IF NOT EXISTS idx_consciousness_history_recorded ON consciousness_state_history(recorded_at DESC);

-- =====================================================
-- Model Weights Storage
-- Versioned neural network weights
-- =====================================================
CREATE TABLE IF NOT EXISTS model_weights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type VARCHAR(50) NOT NULL,
    weights_blob BYTEA NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    architecture_config JSONB,
    training_metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::JSONB,
    CONSTRAINT unique_model_version UNIQUE (model_type, version)
);

CREATE INDEX IF NOT EXISTS idx_model_weights_type ON model_weights(model_type, version DESC);
CREATE INDEX IF NOT EXISTS idx_model_weights_active ON model_weights(model_type, is_active) WHERE is_active;

-- =====================================================
-- Rate Limiting
-- Per-user, per-endpoint rate tracking
-- =====================================================
CREATE TABLE IF NOT EXISTS rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    endpoint VARCHAR(100) NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    CONSTRAINT unique_rate_limit UNIQUE (user_id, endpoint, window_start)
);

CREATE INDEX IF NOT EXISTS idx_rate_limits_user_endpoint ON rate_limits(user_id, endpoint);
CREATE INDEX IF NOT EXISTS idx_rate_limits_window ON rate_limits(window_end);

-- =====================================================
-- Audit Log
-- Track significant operations
-- =====================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50),
    entity_id UUID,
    old_value JSONB,
    new_value JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at DESC);

-- =====================================================
-- Migrations Table (self-tracking)
-- =====================================================
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    execution_time_ms INTEGER,
    checksum VARCHAR(64)
);

-- =====================================================
-- Functions
-- =====================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables with updated_at
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_users_updated_at') THEN
        CREATE TRIGGER trigger_users_updated_at
            BEFORE UPDATE ON users
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_semantic_states_updated_at') THEN
        CREATE TRIGGER trigger_semantic_states_updated_at
            BEFORE UPDATE ON semantic_states
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_working_states_updated_at') THEN
        CREATE TRIGGER trigger_working_states_updated_at
            BEFORE UPDATE ON working_memory_states
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;
END;
$$;

-- Cleanup expired working memory buffers
CREATE OR REPLACE FUNCTION cleanup_expired_working_memory()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM working_memory_buffers WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Cleanup old rate limit entries
CREATE OR REPLACE FUNCTION cleanup_old_rate_limits()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM rate_limits WHERE window_end < NOW() - INTERVAL '1 hour';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Get or create user by external ID
CREATE OR REPLACE FUNCTION get_or_create_user(p_external_id VARCHAR(255))
RETURNS UUID AS $$
DECLARE
    v_user_id UUID;
BEGIN
    -- Try to get existing user
    SELECT id INTO v_user_id FROM users WHERE external_id = p_external_id;

    -- Create if not exists
    IF v_user_id IS NULL THEN
        INSERT INTO users (external_id)
        VALUES (p_external_id)
        RETURNING id INTO v_user_id;
    ELSE
        -- Update last active
        UPDATE users SET last_active_at = NOW() WHERE id = v_user_id;
    END IF;

    RETURN v_user_id;
END;
$$ LANGUAGE plpgsql;

-- Upsert semantic state
CREATE OR REPLACE FUNCTION upsert_semantic_state(
    p_user_id UUID,
    p_state_tensor BYTEA,
    p_tensor_size INTEGER DEFAULT 256
)
RETURNS UUID AS $$
DECLARE
    v_state_id UUID;
BEGIN
    INSERT INTO semantic_states (user_id, state_tensor, tensor_size)
    VALUES (p_user_id, p_state_tensor, p_tensor_size)
    ON CONFLICT (user_id)
    DO UPDATE SET
        state_tensor = EXCLUDED.state_tensor,
        version = semantic_states.version + 1,
        updated_at = NOW()
    RETURNING id INTO v_state_id;

    RETURN v_state_id;
END;
$$ LANGUAGE plpgsql;

-- Upsert working memory state
CREATE OR REPLACE FUNCTION upsert_working_state(
    p_user_id UUID,
    p_state_tensor BYTEA,
    p_tensor_size INTEGER DEFAULT 256
)
RETURNS UUID AS $$
DECLARE
    v_state_id UUID;
BEGIN
    INSERT INTO working_memory_states (user_id, state_tensor, tensor_size)
    VALUES (p_user_id, p_state_tensor, p_tensor_size)
    ON CONFLICT (user_id)
    DO UPDATE SET
        state_tensor = EXCLUDED.state_tensor,
        updated_at = NOW()
    RETURNING id INTO v_state_id;

    RETURN v_state_id;
END;
$$ LANGUAGE plpgsql;

-- Mark experiences as consolidated
CREATE OR REPLACE FUNCTION mark_experiences_consolidated(p_experience_ids UUID[])
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE episodic_experiences
    SET consolidated = TRUE, consolidated_at = NOW()
    WHERE id = ANY(p_experience_ids) AND NOT consolidated;

    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- Views
-- =====================================================

-- User statistics view
CREATE OR REPLACE VIEW user_statistics AS
SELECT
    u.id,
    u.external_id,
    u.created_at,
    u.last_active_at,
    COUNT(DISTINCT ee.id) AS total_experiences,
    COUNT(DISTINCT ee.id) FILTER (WHERE ee.consolidated) AS consolidated_experiences,
    COUNT(DISTINCT ee.id) FILTER (WHERE NOT ee.consolidated) AS pending_experiences,
    AVG(ee.importance) AS avg_importance,
    EXISTS(SELECT 1 FROM semantic_states ss WHERE ss.user_id = u.id) AS has_semantic_state,
    EXISTS(SELECT 1 FROM working_memory_states ws WHERE ws.user_id = u.id) AS has_working_state
FROM users u
LEFT JOIN episodic_experiences ee ON ee.user_id = u.id
GROUP BY u.id, u.external_id, u.created_at, u.last_active_at;

-- Recent consciousness metrics view
CREATE OR REPLACE VIEW recent_consciousness AS
SELECT
    id,
    integrated_information AS phi,
    self_reference_depth,
    temporal_integration,
    causal_density,
    dynamical_complexity,
    consciousness_level,
    computed_at,
    computation_time_ms
FROM consciousness_metrics
ORDER BY computed_at DESC
LIMIT 100;

-- =====================================================
-- Record this migration
-- =====================================================
INSERT INTO schema_migrations (version, checksum)
VALUES ('001_initial_schema', 'sha256:initial')
ON CONFLICT (version) DO NOTHING;

-- =====================================================
-- Grants (adjust roles as needed)
-- =====================================================
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO neuralsleep_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO neuralsleep_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO neuralsleep_user;
