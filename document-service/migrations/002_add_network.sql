-- Migration: 002_add_network.sql
-- Adiciona coluna network_prefix à tabela documents.
-- Rodar após 001_init.sql:
--   psql -U postgres -d docsdb -f migrations/002_add_network.sql

ALTER TABLE documents
  ADD COLUMN IF NOT EXISTS network_prefix TEXT NOT NULL DEFAULT '';

-- Remove a PK simples (user_id) e cria PK composta (user_id + network_prefix)
-- para permitir um documento por usuário por rede.
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_pkey;
ALTER TABLE documents ADD PRIMARY KEY (user_id, network_prefix);