import os
from reconhecimento_facial.src.services.model_processing import ModelProcessing
from reconhecimento_facial.src.model.middleware import Middleware
from reconhecimento_facial.src.model.broker import Broker
from reconhecimento_facial.src.controller.user_interface import User2SInterface

# Caminhos dos modelos
modelFile = "./Modelo/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./Conf/deploy.prototxt"
openface_model_path = './openface/nn4.small2.v1.t7'

# Inicializar componentes
model_processing = ModelProcessing((configFile, modelFile), openface_model_path)
middleware = Middleware(threshold=0.8)
broker = Broker()

# Diretório de imagens de referência e de salvamento
reference_dir = './rostos_conhecidos/'
save_dir = './rostos_detectados/'

# Interface do usuário
user_interface = User2SInterface(model_processing, middleware, broker, reference_dir, save_dir)

# Processar a imagem do usuário

image_dir = "./Imagem/"

# pegar primeira imagem da pasta
images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if len(images) == 0:
    print("Nenhuma imagem encontrada na pasta Imagem.")
else:
    user_image_path = os.path.join(image_dir, images[0])
    print(f"Processando imagem: {user_image_path}")
    user_interface.process_user_image(user_image_path)

