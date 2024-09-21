import os
from model_processing import ModelProcessing
from middleware import Middleware
from broker import Broker
from user_interface import User2SInterface

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
