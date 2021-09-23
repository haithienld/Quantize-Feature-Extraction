from resnet import resnet_tf1 as resnet
from vgg import vgg_tf1 as vgg
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model,load_model
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

wandb.config.update({
    "epochs" : config["epochs"],
    "image_size" : config["image_size"],
    "embedding_size" : config["embedding_size"],
    "num_nodes" : config["num_nodes"],
    "learning_rate" : config["learning_rate"],
    "dropout" : config["dropout"],
    "lossWeights" : config["lossWeights"],
    "steps_per_epoch" : config["steps_per_epoch"],
    "quantize": "yes",
    "quant_delay":  config["quant_delay"]
})

def fix_error():
    gp = tf.get_default_graph().as_graph_def()
    for node in gp.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
                # input0: ref: Should be from a Variable node. May be uninitialized.
                # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]

def full_model(image_size = 48, embedding_size = 1024):
    input_image_shape = (image_size,image_size,3)
    input_images = Input(shape=input_image_shape, name='input_image') # input layer for images
    # base_network = resnet.create_res_net(input_images,embedding_size)
    base_network = vgg.create_vgg(image_size,embedding_size)
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

    return model,base_network,identity

if not os.path.isdir(config["train_folder"]):
    os.makedirs(config["train_folder"])

train_graph = tf.Graph()
train_sess = tf.compat.v1.Session(graph=train_graph)
tf.compat.v1.keras.backend.set_session(train_sess)

path = config["root_folder"]
image_size = config["image_size"]
embedding_size = config["embedding_size"]
data = DataLoader(path,embedding_size,image_size)
train_generator = data.get_batches(train=True)
valid_generator = data.get_batches(train=False)
if config["optimizer"] == "rmsprop":
    optim = RMSprop(lr=config["learning_rate"], rho=0.9, epsilon=1e-08, decay=0.0)
if config["optimizer"] == "adam":
    optim = Adam(lr=config["learning_rate"])

with train_graph.as_default():
    tf.keras.backend.set_learning_phase(1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(config["train_folder"],'checkpoint'),
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=6, min_lr=0.000001)
    model,base_model,identity = full_model(image_size,embedding_size)
    print(model.summary(),identity)
    losses = {
    	identity: center_loss,
    	"classification_output": "binary_crossentropy",
    }
    lossWeights = {identity: config['lossWeights'], "classification_output": 1 - config['lossWeights']}
    tf.contrib.quantize.experimental_create_training_graph(
            input_graph=train_graph,
            quant_delay=config['quant_delay']
        )
    train_sess.run(tf.compat.v1.global_variables_initializer())
    model.compile(optimizer=optim, loss=losses, loss_weights=lossWeights)
    if os.path.isfile(f"{config['train_folder']}/checkpoint"):
        print("load from checkpoint")
        model.load_weights(f"{config['train_folder']}/checkpoint")
    history = model.fit_generator(train_generator,
                        validation_data=valid_generator,
                        epochs=config['epochs'],
                        verbose=2,steps_per_epoch=config["steps_per_epoch"],
                        validation_steps=int(0.3*config["steps_per_epoch"]),
                        callbacks=[reduce_lr,model_checkpoint,WandbCallback()])
    saver = tf.compat.v1.train.Saver()
    saver.save(train_sess, f"{config['train_folder']}/ckpt",global_step=tf.train.get_global_step())
model.save(f"{config['train_folder']}/model.h5")

out = [model.output[i].op.name for i in range(len(model.output))]
eval_graph = tf.Graph()
eval_sess = tf.compat.v1.Session(graph=eval_graph)
tf.keras.backend.set_session(eval_sess)

with eval_graph.as_default():
    tf.keras.backend.set_learning_phase(0)
    model,base_model,identity = full_model(image_size,embedding_size)


    ''' get model '''
    # model.load_weights('model.h5')
    tf.contrib.quantize.experimental_create_eval_graph(
        input_graph=eval_graph
    )
    eval_graph_def = eval_graph.as_graph_def()

    saver = tf.compat.v1.train.Saver()
    saver.restore(eval_sess, f"{config['train_folder']}/ckpt")
    # fix_error()

    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        eval_sess,
        eval_graph_def,
        out
    )

    print("[Train_Data] Model_PB_Path：{}".format('eval_graph_frozen.pb'))

    with open(f"{config['train_folder']}/eval_graph_frozen.pb", 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

print("done")
