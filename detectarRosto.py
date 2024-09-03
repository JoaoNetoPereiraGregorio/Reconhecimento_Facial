import cv2
import numpy as np
import os

# Carregar o modelo de detecção de rostos e o arquivo de configuração
modelFile = "./Modelo/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./Conf/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Carregar o modelo de reconhecimento facial OpenFace
openface_model = cv2.dnn.readNetFromTorch('./openface/nn4.small2.v1.t7')

# Função para extrair o embedding facial usando OpenFace
def get_face_embedding(face_image):
    face_blob = cv2.dnn.blobFromImage(face_image, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    openface_model.setInput(face_blob)
    return openface_model.forward()

# Função para calcular a distância euclidiana
def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

# Carregar a imagem
image = cv2.imread("./Imagem/1.png")
(h, w) = image.shape[:2]

# Pré-processar a imagem para detecção de rostos
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Passar a imagem pela rede para detectar rostos
net.setInput(blob)
detections = net.forward()

# Criar um diretório para salvar as imagens dos rostos detectados
output_dir = "rostos_detectados"
os.makedirs(output_dir, exist_ok=True)

# Variável para indicar se algum rosto foi encontrado
encontrou_match = False

# Diretório que contém as imagens de referência
reference_dir = './rostos_conhecidos/'

# Listar todas as imagens de referência
reference_images = [f for f in os.listdir(reference_dir) if os.path.isfile(os.path.join(reference_dir, f))]

# Iterar sobre as detecções de rostos
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Filtrar as detecções com base na confiança
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Desenhar a caixa delimitadora ao redor do rosto
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Extrair o rosto detectado
        face = image[startY:endY, startX:endX]
        face_resized = cv2.resize(face, (96, 96))  # Redimensionar para 96x96, como exigido pelo OpenFace
        
        # Obter o embedding do rosto detectado
        face_embedding = get_face_embedding(face_resized)

        # Iterar sobre as imagens de referência
        for ref_image_name in reference_images:
            ref_image_path = os.path.join(reference_dir, ref_image_name)
            reference_image = cv2.imread(ref_image_path)
            reference_face = cv2.resize(reference_image, (96, 96))
            reference_embedding = get_face_embedding(reference_face)

            # Calcular a distância euclidiana em relação ao embedding de referência
            distance = euclidean_distance(reference_embedding, face_embedding)

            # Definir um limiar para considerar como um match (ajustar conforme necessário)
            threshold = 0.6

            # Verificar se o rosto detectado corresponde à imagem de referência
            if distance < threshold:
                encontrou_match = True
                print(f"Rosto {i} corresponde à referência '{ref_image_name}' com distância {distance:.4f}")

        # Salvar a imagem do rosto
        face_filename = os.path.join(output_dir, f"face_{i}.jpg")
        cv2.imwrite(face_filename, face)

# Verificar se algum match foi encontrado
if encontrou_match:
    print("Um ou mais rostos conhecidos estão presentes na imagem.")
else:
    print("Nenhum rosto conhecido foi encontrado na imagem.")

