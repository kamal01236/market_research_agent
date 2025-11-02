#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Drop existing tables (if exist)
    DROP TABLE IF EXISTS stocks CASCADE;
    DROP TABLE IF EXISTS features CASCADE;
    DROP TABLE IF EXISTS scores CASCADE;
    
    -- Drop any custom functions
    DROP FUNCTION IF EXISTS calculate_vwap CASCADE;
    DROP FUNCTION IF EXISTS calculate_ma CASCADE;
    
    -- Drop extensions (optional)
    DROP EXTENSION IF EXISTS timescaledb CASCADE;
EOSQL