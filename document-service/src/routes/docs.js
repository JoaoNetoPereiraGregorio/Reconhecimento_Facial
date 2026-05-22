import { Router } from 'express'
import { pool } from '../db.js'

export const router = Router()

/**
 * GET /docs/:user_id
 * Retorna o documento do usuário.
 * Se ainda não existir, cria um documento vazio (upsert lazy).
 */
router.get('/:user_id', async (req, res) => {
  const { user_id } = req.params

  try {
    // Cria o documento vazio na primeira vez que o usuário for reconhecido
    const result = await pool.query(
      `INSERT INTO documents (user_id, text)
       VALUES ($1, '')
       ON CONFLICT (user_id) DO UPDATE SET updated_at = now()
       RETURNING user_id, text, updated_at`,
      [user_id]
    )
    res.json(result.rows[0])
  } catch (err) {
    console.error('GET /docs/:user_id', err)
    res.status(500).json({ error: 'Erro ao buscar documento.' })
  }
})

/**
 * PATCH /docs/:user_id
 * Salva o texto atual do documento (chamado pelo autosave do frontend).
 * Body: { text: string }
 */
router.patch('/:user_id', async (req, res) => {
  const { user_id } = req.params
  const { text } = req.body

  if (typeof text !== 'string') {
    return res.status(400).json({ error: 'Campo "text" obrigatório (string).' })
  }

  try {
    const result = await pool.query(
      `UPDATE documents
       SET text = $1, updated_at = now()
       WHERE user_id = $2
       RETURNING user_id, updated_at`,
      [text, user_id]
    )

    if (result.rowCount === 0) {
      return res.status(404).json({ error: 'Documento não encontrado.' })
    }

    res.json(result.rows[0])
  } catch (err) {
    console.error('PATCH /docs/:user_id', err)
    res.status(500).json({ error: 'Erro ao salvar documento.' })
  }
})