import cv2
import numpy as np

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
