import os
import cv2
from src.model.broker import Broker
from src.model.middleware import Middleware
from src.services.model_processing import ModelProcessing

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
