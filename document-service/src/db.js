import pg from 'pg'
 
export const pool = new pg.Pool({
  host:     'localhost',
  port:     5432,
  database: 'docsdb',
  user:     'postgres',
  password: 'toor', // <-- Escreva a sua senha real aqui diretamente
})