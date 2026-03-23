#!/bin/bash
# setup_db.sh — Initialize PostgreSQL tables for pgvector RAG storage
# Database: bhavya_manya @ 192.240.1.177

set -e

DB_HOST="YOUR_DB_HOST"
DB_NAME="YOUR_DB_NAME"
DB_USER="YOUR_DB_USER"
DB_PASSWORD="YOUR_DB_PASSWORD

echo "[setup_db] Ensuring pgvector extension exists..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
    CREATE EXTENSION IF NOT EXISTS vector;
"

echo "[setup_db] Creating langchain_pg_collection table if not exists..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
    CREATE TABLE IF NOT EXISTS langchain_pg_collection (
        uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name VARCHAR NOT NULL UNIQUE,
        cmetadata JSONB
    );
"

echo "[setup_db] Creating langchain_pg_embedding table if not exists..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
    CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
        embedding vector,
        document TEXT,
        cmetadata JSONB
    );
"

echo "[setup_db] Creating index on embeddings..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
    CREATE INDEX IF NOT EXISTS idx_langchain_pg_embedding_collection
    ON langchain_pg_embedding (collection_id);
"

echo "[setup_db] Verifying tables..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "\dt"

echo "[setup_db] Done! Database is ready for RAG ingestion."
