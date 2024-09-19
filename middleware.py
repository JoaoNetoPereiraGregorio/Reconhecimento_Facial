import numpy as np

class Middleware:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def compare_embeddings(self, reference_embedding, face_embedding):
        """Compara embeddings usando distância euclidiana."""
        distance = np.linalg.norm(reference_embedding - face_embedding)
        return distance < self.threshold, distance
