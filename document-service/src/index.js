import 'dotenv/config'//remover
import express from 'express'
//import cors from 'cors'//remover
import { pool } from './db.js'
import { router as docsRouter } from './routes/docs.js'

const app = express()
app.use(express.json())
//app.use(cors())// remover
app.set('trust proxy', 'loopback, linklocal, uniquelocal');

app.use('/docs', docsRouter)

app.get('/health', async (req, res) => {
  try {
    await pool.query('SELECT 1')
    res.json({ status: 'ok' })
  } catch {
    res.status(503).json({ status: 'db_unavailable' })
  }
})

const PORT = process.env.PORT ?? 8083
app.listen(PORT, '0.0.0.0' ,() => console.log(`document-service :${PORT}`))