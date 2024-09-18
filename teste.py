import cv2
import numpy as np
import os

# ========== 2S Broker ========== 
class Broker:
    @staticmethod
    def log_event(event_message):
        """Loga um evento local."""
        print(f"Evento: {event_message}")
    
    @staticmethod
    def execute_command(command):
        """Executa um comando (por exemplo, notificar usuário)."""
        print(f"Comando executado: {command}")

# ========== User-Centric 2S Middleware ========== 
class Middleware:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def compare_embeddings(self, reference_embedding, face_embedding):
        """Compara embeddings usando distância euclidiana."""
        distance = np.linalg.norm(reference_embedding - face_embedding)
        return distance < self.threshold, distance

# ========== Model Processing ========== 
class ModelProcessing:
    def __init__(self, face_detection_model, openface_model_path):
        # Carregar o modelo de detecção de rostos
        self.face_net = cv2.dnn.readNetFromCaffe(*face_detection_model)
        # Carregar o modelo de reconhecimento facial OpenFace
        self.openface_model = cv2.dnn.readNetFromTorch(openface_model_path)

    def detect_faces(self, image):
        """Detecta rostos em uma imagem."""
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                faces.append(face)
        return faces

    def get_face_embedding(self, face_image):
        """Extrai o embedding facial usando OpenFace."""
        face_blob = cv2.dnn.blobFromImage(face_image, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.openface_model.setInput(face_blob)
        return self.openface_model.forward()

# ========== User-2S Interface ========== 
class User2SInterface:
    def __init__(self, model_processing, middleware, broker, reference_dir, save_dir):
        self.model_processing = model_processing
        self.middleware = middleware
        self.broker = broker
        self.reference_dir = reference_dir
        self.save_dir = save_dir

    def load_reference_images(self):
        """Carrega as imagens de referência e seus embeddings."""
        reference_images = {}
        for ref_image_name in os.listdir(self.reference_dir):
            ref_image_path = os.path.join(self.reference_dir, ref_image_name)
            if os.path.isfile(ref_image_path):
                reference_image = cv2.imread(ref_image_path)
                reference_face = cv2.resize(reference_image, (96, 96))
                reference_embedding = self.model_processing.get_face_embedding(reference_face)
                reference_images[ref_image_name] = reference_embedding
        return reference_images

    def save_detected_face(self, face, face_id):
        """Salva o rosto detectado em um arquivo de imagem."""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        face_path = os.path.join(self.save_dir, f"face_{face_id}.png")
        cv2.imwrite(face_path, face)
        self.broker.log_event(f"Rosto {face_id} salvo em {face_path}.")

    def process_user_image(self, user_image_path):
        """Processa a imagem do usuário e compara com as referências."""
        # Carregar a imagem do usuário
        image = cv2.imread(user_image_path)
        
        # Detectar rostos
        faces = self.model_processing.detect_faces(image)
        
        if not faces:
            self.broker.log_event("Nenhum rosto detectado.")
            return

        # Carregar imagens de referência
        reference_images = self.load_reference_images()

        encontrou_match = False

        # Processar cada rosto detectado
        for i, face in enumerate(faces):
            face_resized = cv2.resize(face, (96, 96))
            face_embedding = self.model_processing.get_face_embedding(face_resized)

            # Comparar com cada referência
            for ref_name, ref_embedding in reference_images.items():
                match, distance = self.middleware.compare_embeddings(ref_embedding, face_embedding)
                if match:
                    encontrou_match = True
                    self.broker.execute_command(f"Rosto {i} corresponde à referência '{ref_name}' com distância {distance:.4f}")

            # Salvar o rosto detectado
            self.save_detected_face(face, i)

        # Verificar se algum rosto foi reconhecido
        if encontrou_match:
            self.broker.log_event("Um ou mais rostos conhecidos foram encontrados.")
        else:
            self.broker.log_event("Nenhum rosto conhecido foi encontrado.")

# ========== Execução Principal ========== 
if __name__ == "__main__":
    # Caminhos dos modelos
    modelFile = "./Modelo/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "./Conf/deploy.prototxt"
    openface_model_path = './openface/nn4.small2.v1.t7'
    
    # Inicializar componentes
    model_processing = ModelProcessing((configFile, modelFile), openface_model_path)
    middleware = Middleware(threshold=0.6)
    broker = Broker()
    
    # Diretório de imagens de referência e de salvamento
    reference_dir = './rostos_conhecidos/'
    save_dir = './rostos_detectados/'
    
    # Interface do usuário
    user_interface = User2SInterface(model_processing, middleware, broker, reference_dir, save_dir)
    
    # Processar a imagem do usuário
    user_image_path = "./Imagem/1.png"
    user_interface.process_user_image(user_image_path)
