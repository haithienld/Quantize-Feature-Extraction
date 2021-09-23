import numpy as np
from resnet import resnet_quantize_v2 as resnet_quantize
from resnet import resnet
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from data.image_preprocess import preprocess
import cv2
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--first", default=str)
parser.add_argument("--second", default=str)
img = parser.parse_args()
with open('config.json', 'r') as f:
    config = json.load(f)
path = config["root_folder"]
image_size = config["image_size"]
embedding_size = config["embedding_size"]
input_image_shape = (image_size,image_size,3)
# base_network = resnet_quantize.create_res_net_quantize(input_image_shape,embedding_size)
"""/////////////////////////////"""
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
from loss.center_loss_classification import center_loss
import os
def full_model(image_size = 48, embedding_size = 1024,quantize=False):
    input_image_shape = (image_size,image_size,3)
    if quantize == True:
        base_network = resnet_quantize.create_res_net_quantize(input_image_shape,embedding_size)
    else:
        base_network = resnet.create_res_net(input_image_shape,embedding_size)
        # base_network.load_weights('base_model.h5')
#        base_network.set_weights(weights)
    input_images = Input(shape=input_image_shape, name='input_image') # input layer for images
    embeddings = base_network([input_images])              # output of network -> embeddings
    identity = embeddings.name.split("/")[0]
    # outputs = Dropout(config["dropout"])(embeddings)
    # outputs = Dense(config["num_nodes"])(outputs)
    # outputs = Dropout(config["dropout"])(outputs)
    outputs = Dense(2,activation="sigmoid",name="classification_output")(embeddings)
    model = Model(inputs=input_images,
                          outputs=[embeddings,outputs])

    # base_network.summary()
    return model,base_network,identity

model,base_model,identity = full_model(image_size,embedding_size,quantize=False)
model.load_weights('checkpoint/tempt_224/checkpoint')
layer = model.get_layer(identity)
weights = layer.get_weights()
base_model.set_weights(weights)

"""///////////////////////////////////"""
# base_network.load_weights('base_model_quant.h5')
print(os.path.join(os.path.join(path,'image'),img.first))
first = preprocess(os.path.join(os.path.join(path,'image'),img.first),image_size,single=True)
second = preprocess(os.path.join(os.path.join(path,'image'),img.second),image_size,single=True)
f1 = base_model.predict(first)
f2 = base_model.predict(second)
k = np.stack([f1[0],f2[0],f1[0]])
print(k.shape)
different = np.sum((f1 - k)**2,axis=1)
print(different)
