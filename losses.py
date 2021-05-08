from keras import backend as K
import numpy as np
import tensorflow as tf

def focal_loss(targets, inputs, alpha=0.8, gamma=2):    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    return focal_loss

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)

def weighted_dice_loss(y_true, y_pred,weights):
    weight = np.array([1-0.008129217,1-0.741332343,1-0.038759669,1-0.033971285,1-0.159327414,1-0.018480072])
    smooth = 1.
    w, m1, m2 = weights * weights, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * tf.reduce_sum(w * intersection) + smooth) / \
            (tf.reduce_sum(w * m1) + tf.reduce_sum(w * m2) + smooth)
    print(score)
    loss = 1. - tf.reduce_sum(score)
    return loss

def jaccard_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(
        tf.keras.backend.abs(y_true * y_pred), axis=-1)
    sum_ = tf.keras.backend.sum(tf.keras.backend.abs(
        y_true) + tf.keras.backend.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac * smooth