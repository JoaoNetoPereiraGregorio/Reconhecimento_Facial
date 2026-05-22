import express from 'express'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

const app = express()
const PORT = 8081

// Configuração para descobrir o caminho da pasta atual usando ES Modules
const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// Serve todos os arquivos da pasta atual (index.html, imagens, css, js)
app.use(express.static(__dirname))

// Rota principal que entrega o seu index.html
app.get('/', (req, res) => {
  res.sendFile(join(__dirname, 'index.html'))
})

app.listen(PORT,'0.0.0.0', () => {
  console.log(`💻 Frontend rodando em: http://localhost:${PORT}`)
})
