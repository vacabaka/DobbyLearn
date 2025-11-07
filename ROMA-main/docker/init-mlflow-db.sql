-- Initialize MLflow database for PostgreSQL backend
-- This script runs automatically when PostgreSQL container starts

-- Create mlflow database if it doesn't exist
SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

-- Grant all privileges to the postgres user on mlflow database
GRANT ALL PRIVILEGES ON DATABASE mlflow TO postgres;
