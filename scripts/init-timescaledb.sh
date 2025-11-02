#!/bin/bash
set -e

# Initialize TimescaleDB extension
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
    
    -- Create stocks table
    CREATE TABLE IF NOT EXISTS stocks (
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        open NUMERIC NOT NULL,
        high NUMERIC NOT NULL,
        low NUMERIC NOT NULL,
        close NUMERIC NOT NULL,
        volume BIGINT NOT NULL,
        PRIMARY KEY (symbol, timestamp)
    );
    
    -- Convert stocks to hypertable
    SELECT create_hypertable('stocks', 'timestamp', if_not_exists => TRUE);
    
    -- Create features table
    CREATE TABLE IF NOT EXISTS features (
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        feature_name VARCHAR(50) NOT NULL,
        feature_value NUMERIC NOT NULL,
        PRIMARY KEY (symbol, timestamp, feature_name)
    );
    
    -- Convert features to hypertable
    SELECT create_hypertable('features', 'timestamp', if_not_exists => TRUE);
    
    -- Create scores table
    CREATE TABLE IF NOT EXISTS scores (
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        factor_name VARCHAR(50) NOT NULL,
        score NUMERIC NOT NULL,
        PRIMARY KEY (symbol, timestamp, factor_name)
    );
    
    -- Convert scores to hypertable
    SELECT create_hypertable('scores', 'timestamp', if_not_exists => TRUE);
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks (symbol);
    CREATE INDEX IF NOT EXISTS idx_stocks_timestamp ON stocks (timestamp DESC);
    
    CREATE INDEX IF NOT EXISTS idx_features_symbol ON features (symbol);
    CREATE INDEX IF NOT EXISTS idx_features_timestamp ON features (timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_features_name ON features (feature_name);
    
    CREATE INDEX IF NOT EXISTS idx_scores_symbol ON scores (symbol);
    CREATE INDEX IF NOT EXISTS idx_scores_timestamp ON scores (timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_scores_factor ON scores (factor_name);
EOSQL