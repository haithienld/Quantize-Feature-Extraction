from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU,BatchNormalization,MaxPool2D,\
                                    Add, AveragePooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
# from tensorflow.layers import BatchNormalization
import tensorflow as tf



def relu_bn(inputs: Tensor) -> Tensor:
    bn = BatchNormalization(fused=False)(inputs)
    # relu = ReLU()(bn)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same",use_bias=False,activation='relu')(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same",use_bias=False,activation='relu')(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same",use_bias=False,activation='relu')(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net(inputs,embedding_size):
    num_filters = 64

    # t = BatchNormalization(fused=False)(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same",use_bias=False,activation='relu')(inputs)
    # t = BatchNormalization(fused=False)(t)
    t = relu_bn(t)

    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        if i >=1 and i<3:
            t = MaxPool2D((2,2))(t)
        num_filters *= 2

    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(embedding_size,use_bias=False)(t)
    seg1 = tf.identity(Dense(int(embedding_size/4))(t),name='seg1')
    seg2 = tf.identity(Dense(int(embedding_size/4))(t),name='seg2')
    seg3 = tf.identity(Dense(int(embedding_size/4))(t),name='seg3')
    seg4 = tf.identity(Dense(int(embedding_size/4))(t),name='seg4')

    outputs = Concatenate()([seg1, seg2, seg3,seg4])
    model = Model(inputs, outputs)

    return model
