from resnet import resnet_quantize as resnet_quantize
from resnet import resnet
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
from loss.center_loss_classification import center_loss
from data.load_data import  DataLoader
import logging
import os
import json
import wandb
from wandb.keras import WandbCallback

wandb.init(project="vehicles_features_extractor")
with open('config.json', 'r') as f:
    config = json.load(f)

logging.basicConfig(level = logging.INFO)

wandb.config.update({
    "epochs" : config["epochs"],
    "image_size" : config["image_size"],
    "embedding_size" : config["embedding_size"],
    "num_nodes" : config["num_nodes"],
    "learning_rate" : config["learning_rate"],
    "dropout" : config["dropout"],
    "lossWeights" : config["lossWeights"],
    "steps_per_epoch" : config["steps_per_epoch"],
    "quantize": "yes"
})

def full_model(image_size = 48, embedding_size = 1024,quantize=False):
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
    # outputs = Dense(config["num_nodes"],activation="relu")(outputs)
    # outputs = Dropout(config["dropout"])(outputs)
    # outputs = Dense(128,activation="relu")(outputs)
    # outputs = Dropout(config["dropout"])(outputs)
    outputs = Dense(2,activation="sigmoid",name="classification_output")(embeddings)

    model = Model(inputs=input_images,
                          outputs=[embeddings,outputs])

    # model.summary()
    # base_network.summary()
    return model,base_network,identity


logging.info("...CREATING GENERATORS...\n")
# path = '/home/adamduong26111996/shared_data/Vehicle-1M/'
# path = '/media/dtlam26/c1f130b0-9ac8-4f32-94bd-1806f3b37a8f/dtlam26/Documents/data/Vehicle-1M/Vehicle-1M'
path = config["root_folder"]
image_size = config["image_size"]
embedding_size = config["embedding_size"]

if not os.path.isdir(os.path.join(config["train_folder"])):
    os.makedirs(config["train_folder"])
checkpoint_filepath = os.path.join(config["train_folder"],'checkpoint_quant/checkpoint')
if not os.path.isdir(os.path.join(config["train_folder"],'checkpoint_quant')):
    os.makedirs(os.path.join(config["train_folder"],'checkpoint_quant'))
if not os.path.isdir(os.path.join(config["train_folder"],'logs_quant')):
    os.makedirs(os.path.join(config["train_folder"],'logs_quant'))

data = DataLoader(path,embedding_size,image_size)
train_generator = data.get_batches()
valid_generator = data.get_batches()

logging.info("...COMPLETE GENERATORS...\n")

if config["optimizer"] == "rmsprop":
    optim = RMSprop(lr=config["learning_rate"], rho=0.9, epsilon=1e-08, decay=0.0)
if config["optimizer"] == "adam":
    optim = Adam(lr=config["learning_rate"])


model,base_model,identity = full_model(image_size,embedding_size,quantize=True)
model.load_weights(os.path.join(config["train_folder"],'checkpoint/checkpoint'))
# layer = model.get_layer(identity)
# print(layer.summary())
# weights = layer.get_weights()


losses = {
	identity: center_loss,
	"classification_output": "binary_crossentropy",
}
lossWeights = {identity: config['lossWeights'], "classification_output": 1-config['lossWeights']}
# initialize the optimizer and compile the model
logging.info("...COMPILING MODEL & CALLBACKS...\n")
model.compile(optimizer=optim, loss=losses, loss_weights=lossWeights)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=6, min_lr=0.000001)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss:', min_delta=0, patience=8, verbose=1, mode='min',
    baseline=None, restore_best_weights=True
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(config["train_folder"],'logs_quant'))

callbacks = list()
for call in config["callbacks"]:
    if call == "early_stopping":
        callbacks.append(early_stopping)
    if call == "reduce_lr":
        callbacks.append(reduce_lr)
    # if call == "tb":
    #     callbacks.append(tensorboard_callback)
callbacks.append(model_checkpoint)
callbacks.append(WandbCallback())

logging.info("...TRAINING...\n")
history = model.fit_generator(train_generator,
                    validation_data=valid_generator,
                    epochs=10,
                    verbose=2,steps_per_epoch=config["steps_per_epoch"],
                    validation_steps=int(0.3*config["steps_per_epoch"]),
                    callbacks=callbacks)

logging.info("...SAVE MODELS...\n")
model.save(f"{os.path.join(config['train_folder'],'model_quant.h5')}")
