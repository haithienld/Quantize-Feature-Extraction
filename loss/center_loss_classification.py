import tensorflow as tf
import numpy as np
import warnings


# def center_loss(y_true,y_pred):
#     del y_true
#     try:
#         pos = y_pred[:5, :]
#         hard = y_pred[5:10,:]
#         other = y_pred[10:,:]
#     except:
#         print(y_pred)
#     anchor = tf.reduce_mean(pos, axis=0)
#     pos_dist = tf.reduce_sum(tf.square(anchor-pos),axis=1)
#     neg_dist1 = tf.reduce_sum(tf.square(anchor-hard),axis=1)
#     neg_dist2 = tf.reduce_sum(tf.square(anchor-other),axis=1)
#     loss1 = tf.maximum(pos_dist-neg_dist1+0.4,0.0)
#     loss2 = tf.maximum(pos_dist-neg_dist1+0.2,0.0)
#     loss = 0.5*loss1+0.5*loss2
#     # loss = tf.reduce_mean(loss)
#     return loss


def center_loss(y_true,y_pred):
    del y_true
    try:
        pos1 = y_pred[:5, :]
        pos2 = y_pred[5:10, :]
        hard = y_pred[10:15,:]
        other = y_pred[15:,:]
    except:
        print(y_pred)
    anchor1 = tf.reduce_mean(pos1, axis=0)
    pos_dist = tf.reduce_sum(tf.square(anchor1-pos1),axis=1)
    neg_dist1 = tf.reduce_sum(tf.square(anchor1-hard),axis=1)
    neg_dist2 = tf.reduce_sum(tf.square(anchor1-other),axis=1)
    loss1 = tf.maximum(pos_dist-neg_dist1+0.4,0.0)
    loss2 = tf.maximum(pos_dist-neg_dist1+0.2,0.0)
    loss_g1 = 0.5*loss1+0.5*loss2

    anchor2 = tf.reduce_mean(pos2, axis=0)
    pos_dist = tf.reduce_sum(tf.square(anchor2-pos2),axis=1)
    neg_dist1 = tf.reduce_sum(tf.square(anchor2-hard),axis=1)
    neg_dist2 = tf.reduce_sum(tf.square(anchor2-other),axis=1)
    loss1 = tf.maximum(pos_dist-neg_dist1+0.4,0.0)
    loss2 = tf.maximum(pos_dist-neg_dist1+0.2,0.0)
    loss_g2 = 0.5*loss1+0.5*loss2

    pos_dist1 = tf.reduce_sum(tf.square(anchor1-pos2),axis=1)
    pos_dist2 = tf.reduce_sum(tf.square(anchor2-pos1),axis=1)
    loss_truth = tf.maximum(pos_dist1+pos_dist2,0.0005)
    loss = loss_g1+loss_g2+loss_truth
    return loss
