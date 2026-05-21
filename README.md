# Reconhecimento Facial

Microserviço leve de reconhecimento facial usando OpenCV e embeddings no estilo OpenFace.

## O que faz
- Detecta rostos em imagens enviadas
- Extrai embeddings faciais e compara com uma pasta de rostos conhecidos
- Retorna JSON com resultados de correspondência e registra eventos úteis no terminal

## Funcionalidades
- API REST: `/recognize`, `/health`, `/reload-references`
- Salvamento opcional de uploads com nomes de arquivo seguros contra colisões (`./uploads/`)
- Arquitetura simples: `ModelProcessing`, `Middleware`, `Broker`, `User2SInterface`

## Pré-requisitos
- Python

## Instalação rápida
```bash
pip install flask opencv-python numpy #instala os pre requisitos do codigo
python app.py #roda o arquivo principal
```

O serviço escuta na porta `5000`.

## Endpoints
- `GET /health` — retorna status do serviço e quantidade de referências carregadas
- `POST /recognize` — funcionalidade principal de detecção facial, usa uma imagem para iniciar.
- `POST /reload-references` — recarrega `rostos_conhecidos/` sem reiniciar o sistema.

Exemplo `curl` para testar reconhecimento:

```bash
curl -X POST http://localhost:5000/recognize -F "image=@C:/caminho/para/foto.jpg"
```

## Usando Postman / Navegador
- Postman: abra uma aba `POST`, `http://localhost:5000/recognize` , vá para aba `Body` → troque de `none` para `form-data` → nomeie o primeiro campo de `image` → troque de `text` para `File` → escolha a imagem em disco → `Send`

## Imagens de referência
- Coloque uma imagem por pessoa em `rostos_conhecidos/` (melhor: foto frontal, boa iluminação, imitando o posicionamento de uma webcam fixa)
- Após adicionar imagens, chame `POST /reload-references` ou reinicie o serviço para que sejam usadas pelo sistema.


## Licença & Contato
Projeto pessoal/demonstração.
