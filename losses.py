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

def dice_coef_loss():
    def loss_func(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)
    return loss_func
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

def weighted_dice_loss(weights):
    def loss_func(y_true, y_pred):
        smooth = 1.
        w, m1, m2 = weights * weights, y_true, y_pred
        intersection = (m1 * m2)
        score = (2. * tf.reduce_sum(w * intersection) + smooth) / \
        (tf.reduce_sum(w * m1) + tf.reduce_sum(w * m2) + smooth)
        print(score)
        loss_value = 1. - tf.reduce_sum(score)
        return loss_value
    return loss_func

def jaccard_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(
        tf.keras.backend.abs(y_true * y_pred), axis=-1)
    sum_ = tf.keras.backend.sum(tf.keras.backend.abs(
        y_true) + tf.keras.backend.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac * smooth

""" define mean iou accuracy maetric """
objects_id = {
    "ball_iou"     : 0,
    "field_iou"    : 1,
    "robot_iou"    : 2,
    "line_iou"     : 3,
    "back_gnd_iou" : 4,
    "goal_iou"     : 5
}

class object_mean_iou(tf.keras.metrics.Metric):
    def __init__(self, name='mean_iou', **kwargs):
        super(object_mean_iou, self).__init__(name=name, **kwargs)
        # self.name = name
        self.intersection = 0.0
        self.union = 0.0
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_1D_map = tf.argmax(y_true, axis=-1)
        y_pred_1D_map = tf.argmax(y_pred, axis=-1)
        true_class_map = tf.cast(tf.equal(y_true_1D_map, objects_id[self.name]),dtype=tf.int32)
        pred_class_map = tf.cast(tf.equal(y_pred_1D_map, objects_id[self.name]),dtype=tf.int32)
        self.intersection = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_and(true_class_map,pred_class_map)),tf.float32)
        self.union = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_or(true_class_map,pred_class_map)),tf.float32)
    def result(self):
        return self.intersection/self.union
class total_mean_iou(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)