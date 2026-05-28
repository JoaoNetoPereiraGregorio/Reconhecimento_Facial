import { Router } from 'express'
import { pool } from '../db.js'

export const router = Router()

function clientNetwork(req) {
  let ip = req.headers['x-client-ip'] || req.ip;

  if (!ip) return '';

  ip = ip.replace(/^::ffff:/, '');

  const parts = ip.split('.');
  if (parts.length === 4) return parts.slice(0, 3).join('.');

  return ip;
}





/**
 * GET /docs/:user_id
 * Busca ou cria o documento do usuário para a rede detectada.
 */
router.get('/:user_id', async (req, res) => {
  const { user_id }      = req.params
  const network_prefix   = clientNetwork(req)

  console.log(`GET /docs/${user_id} | ip: ${req.headers['x-real-ip']} | prefix: ${network_prefix}`)

  try {
    const result = await pool.query(
      `INSERT INTO documents (user_id, network_prefix, text)
       VALUES ($1, $2, '')
       ON CONFLICT (user_id, network_prefix) DO UPDATE
         SET updated_at = now()
       RETURNING user_id, network_prefix, text, updated_at`,
      [user_id, network_prefix]
    )
    res.json(result.rows[0])
  } catch (err) {
    console.error('GET /docs/:user_id', err)
    res.status(500).json({ error: 'Erro ao buscar documento.' })
  }
})

/**
 * PATCH /docs/:user_id
 * Salva o texto para o par (user_id, network_prefix).
 * Body: { text: string }
 */
router.patch('/:user_id', async (req, res) => {
  const { user_id }    = req.params
  const { text }       = req.body
  const network_prefix = clientNetwork(req)

  if (typeof text !== 'string') {
    return res.status(400).json({ error: 'Campo "text" obrigatório (string).' })
  }

  console.log(`PATCH /docs/${user_id} | prefix: ${network_prefix}`)

  try {
    const result = await pool.query(
      `UPDATE documents
       SET text = $1, updated_at = now()
       WHERE user_id = $2 AND network_prefix = $3
       RETURNING user_id, network_prefix, updated_at`,
      [text, user_id, network_prefix]
    )

    if (result.rowCount === 0) {
      return res.status(404).json({ error: 'Documento não encontrado para esta rede.' })
    }

    res.json(result.rows[0])
  } catch (err) {
    console.error('PATCH /docs/:user_id', err)
    res.status(500).json({ error: 'Erro ao salvar documento.' })
  }
})