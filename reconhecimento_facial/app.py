"""
Microserviço de Reconhecimento Facial
Endpoint: POST /recognize
Input:  multipart/form-data com campo 'image' (arquivo de imagem)
Output: { "recognized": true/false, "details": [...] }
"""

import os
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Reutiliza as camadas que você já possui
# ---------------------------------------------------------------------------
from src.services.model_processing import ModelProcessing
from src.services.upload_service import save_upload_file
from src.model.middleware import Middleware
from src.model.broker import Broker
from src.controller.user_interface import User2SInterface

# ---------------------------------------------------------------------------
# Configuração dos modelos (ajuste os caminhos conforme seu ambiente)
# ---------------------------------------------------------------------------
MODEL_FILE      = os.getenv("MODEL_FILE",      "./Modelo/res10_300x300_ssd_iter_140000.caffemodel")
CONFIG_FILE     = os.getenv("CONFIG_FILE",     "./Conf/deploy.prototxt")
OPENFACE_MODEL  = os.getenv("OPENFACE_MODEL",  "./openface/nn4.small2.v1.t7")
REFERENCE_DIR   = os.getenv("REFERENCE_DIR",   "./rostos_conhecidos/")
SAVE_DIR        = os.getenv("SAVE_DIR",        "./rostos_detectados/")
THRESHOLD       = float(os.getenv("THRESHOLD", "0.8"))

# ---------------------------------------------------------------------------
# Inicialização única dos componentes pesados (carregados 1x na subida)
# ---------------------------------------------------------------------------
model_processing = ModelProcessing((CONFIG_FILE, MODEL_FILE), OPENFACE_MODEL)
middleware       = Middleware(threshold=THRESHOLD)
broker           = Broker()
user_interface   = User2SInterface(
    model_processing, middleware, broker, REFERENCE_DIR, SAVE_DIR
)

# Pré-carrega embeddings de referência uma única vez
reference_images = user_interface.load_reference_images()

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:8081", "http://127.0.0.1:8081"],
        "methods": ["POST", "GET", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Endpoint de saúde — útil para orquestadores como K8s/Docker Compose."""
    broker.log_event(f"[{request.remote_addr}] GET /health")
    return jsonify({"status": "ok", "references_loaded": len(reference_images)}), 200
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
@app.route("/recognize", methods=["POST"])
def recognize():
    """
    Recebe uma imagem e retorna se algum rosto reconhecido foi encontrado.

    Exemplo de chamada:
        curl -X POST http://localhost:5000/recognize \
             -F "image=@/caminho/para/foto.jpg"

    Resposta de sucesso:
        {
          "recognized": true,
          "matches": [
            { "face_index": 0, "reference": "joao.jpg", "distance": 0.4321 }
          ]
        }

    Resposta sem match:
        { "recognized": false, "matches": [] }
    """
    if "image" not in request.files:
        return jsonify({"error": "Campo 'image' ausente no request."}), 400

    file = request.files["image"]
    if file.filename == "":
        broker.log_event(f"[{request.remote_addr}] POST /recognize - request vazio recebido (arquivo sem nome???)")
        return jsonify({"error": "Houve um erro com o arquivo enviado (arquivo sem nome???)."}), 400

    broker.log_event(f"[{request.remote_addr}] POST /recognize - arquivo recebido: {file.filename}")

    # Salva o arquivo recebido do /post
    # Comente essa parte p n manter histórico dos uploads.
    upload_path, upload_filename, renamed = save_upload_file(file)
    if renamed:
        broker.log_event(f"[{request.remote_addr}] UPLOAD COM NOME JÁ EXISTENTE, atribuido novo nome: {upload_filename}")
    broker.log_event(f"[{request.remote_addr}] Upload salvo em: {upload_path}")
    file.stream.seek(0)  # Retorna o ponteiro do arquivo ao início para leitura em memória

    # Decodifica o arquivo em memória
    file_bytes  = np.frombuffer(file.read(), np.uint8)
    image       = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        broker.log_event(f"[{request.remote_addr}] POST /recognize - falha ao decodificar imagem {file.filename}")
        return jsonify({"error": "Não foi possível decodificar a imagem."}), 422

    # Detecta rostos na imagem recebida
    faces = model_processing.detect_faces(image)

    if not faces:
        broker.log_event(f"[{request.remote_addr}] POST /recognize - nenhum rosto detectado no upload {file.filename}")
        return jsonify({"recognized": False, "matches": []}), 200

    matches     = []
    recognized  = False

    for i, face in enumerate(faces):
        face_resized    = cv2.resize(face, (96, 96))
        face_embedding  = model_processing.get_face_embedding(face_resized)

        for ref_name, ref_embedding in reference_images.items():
            match, distance = middleware.compare_embeddings(ref_embedding, face_embedding)
            if match:
                recognized = True
                user_id = os.path.splitext(ref_name)[0] 
                matches.append({
                    "face_index": i,
                    "user_id":    user_id,
                    "reference":  ref_name,
                    "distance":   round(float(distance), 4),
                })
                broker.log_event(
                    f"[{request.remote_addr}] POST /recognize - match face {i} -> ref={ref_name} user_id={user_id} dist={distance:.4f}"
                )

    if recognized:
        broker.log_event(f"[{request.remote_addr}] POST /recognize - reconhecimento concluído: {len(matches)} match(es) em {len(faces)} rosto(s)")
    else:
        broker.log_event(f"[{request.remote_addr}] POST /recognize - nenhum rosto conhecido encontrado entre {len(faces)} rosto(s) detectado(s)")

    return jsonify({"recognized": recognized, "matches": matches}), 200
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
@app.route("/reload-references", methods=["POST"])
def reload_references():
    """
    Recarrega os embeddings de referência sem reiniciar o serviço.
    Útil quando novos rostos são adicionados à pasta de referências.
    """
    global reference_images
    broker.log_event(f"[{request.remote_addr}] POST /reload-references - recarregando {len(reference_images)} referências")
    reference_images = user_interface.load_reference_images()
    broker.log_event(f"Referências recarregadas: {len(reference_images)} imagens.")
    return jsonify({"reloaded": len(reference_images)}), 200


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # debug=False em produção; use Gunicorn ou uWSGI
    app.run(host="0.0.0.0", port=port, debug=False)