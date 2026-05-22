-- Migration: 001_init.sql
-- Rodar uma vez antes de subir o serviço:
--   psql -U postgres -d docsdb -f migrations/001_init.sql

CREATE TABLE IF NOT EXISTS users (
  user_id    TEXT        PRIMARY KEY,          -- mesmo valor retornado pelo face service
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
  user_id    TEXT        PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
  text       TEXT        NOT NULL DEFAULT '',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Seed de exemplo (opcional — remova em produção)
-- INSERT INTO users (user_id) VALUES ('joao') ON CONFLICT DO NOTHING;