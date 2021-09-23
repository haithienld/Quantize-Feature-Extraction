import numpy as np
from data.load_data import  DataLoader
from resnet import resnet_quantize as resnet_quantize
from resnet import resnet
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
from loss.center_loss_classification import center_loss
import os

def full_model(image_size = 48, embedding_size = 1024,quantize=True):
    input_image_shape = (image_size,image_size,3)
    input_images = Input(shape=input_image_shape, name='input_image') # input layer for images
    if quantize == True:
        base_network = resnet_quantize.create_res_net_quantize(input_images,embedding_size,quantize)
    else:
        base_network = resnet.create_res_net(input_images,embedding_size)
        # base_network.load_weights('base_model.h5')
#        base_network.set_weights(weights)
    embeddings = base_network([input_images])              # output of network -> embeddings
    identity = embeddings.name.split("/")[0]
    # outputs = Dropout(config["dropout"])(embeddings)
    # outputs = Dense(config["num_nodes"])(outputs)
    # outputs = Dropout(config["dropout"])(outputs)
    outputs = Dense(2,activation="sigmoid",name="classification_output")(embeddings)
    model = Model(inputs=input_images,
                          outputs=[embeddings,outputs])

    # model.summary()
    # base_network.summary()
    return model,base_network,identity

import tensorflow_model_optimization as tfmot

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope


class DefaultBNQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        # return []
        return [(layer.weights[i], LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False)) for i in range(2)]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
#       # Add this line for each item returned in `get_weights_and_quantizers`
#       # , in the same order
        # layer.kernel = quantize_weights[0]
        # print(quantize_weights)
        layer.gamma = quantize_weights[0]
        layer.beta = quantize_weights[1]
        # layer.moving_mean = quantize_weights[2]
        # layer.moving_variance = quantize_weights[3]
        # pass

    def set_quantize_activations(self, layer, quantize_activations):
      # Add this line for each item returned in `get_activations_and_quantizers`
      # , in the same order.
        pass

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}
import json
with open('config.json', 'r') as f:
    config = json.load(f)

image_size = config["image_size"]
embedding_size = config["embedding_size"]
model,base_model,identity = full_model(image_size,embedding_size,quantize=True)
input_image_shape = (image_size,image_size,3)

# if config["quantize_from"] == "h5":
# model.load_weights(os.path.join(config["train_folder"],'model_quant.h5'))
model.load_weights('resnet_224_v2/model_quant.h5')
#
# if config["quantize_from"] == "ckpt":
# model.load_weights('checkpoint/tempt_224/checkpoint')
# model.load_weights(os.path.join(config["train_folder"],'checkpoint_quant/checkpoint'))
layer = model.get_layer(identity)
# layer.summary()
# weights = layer.get_weights()
# base_model.set_weights(weights)
# base_model = resnet_quantize.create_res_net_quantize(input_image_shape,embedding_size)
# model.load_weights("resnet_224_v1/checkpoint/checkpoint")
# layer = model.get_layer(identity)
path = config["root_folder"]
data = DataLoader(path,embedding_size,image_size)
train_generator = data.get_batches()

print("PERFORM CONVERSION")
converter = tf.lite.TFLiteConverter.from_keras_model(layer)
def representative_dataset_gen():
    for _ in range(50):
        batch = next(train_generator)
        img = np.expand_dims(batch[0][0],0).astype(np.float32)
    # # Get sample input data as a numpy array in a method of your choosing.
        yield [img]
        # yield [np.zeros((1,224,224,3), dtype=np.float32)]
with tfmot.quantization.keras.quantize_scope({'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig}):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.experimental_new_converter = True

    # converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    quantized_tflite_model = converter.convert()
    with open("test.tflite", 'wb') as f:
        f.write(quantized_tflite_model)
# import pathlib
# tflite_model_quant_file = "./test.tflite"
# tflite_model_quant_file.write_bytes(quantized_tflite_model)
