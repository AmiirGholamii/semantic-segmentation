from keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

def focal_loss_1(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=1):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = prediction_tensor
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)



def focal_loss(y_true, y_pred, alpha=0.8, gamma=0):    
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    BCE = K.binary_crossentropy(y_true, y_pred)
    BCE_EXP = K.exp(BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    return focal_loss

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def multi_category_focal_loss1(y_true, y_pred):
    epsilon = 1.e-7
    # first reduce along the first dimention (dimention 0 == all one-hot lables in input batch)
    # then reduce along the second and third dimentions (dimention 1,2 = rows ,columns dimentions)
    class_count = tf.reduce_sum(y_true,[0,1,2]) + 1
    class_weights = tf.reduce_sum(class_count) /  class_count
    class_weights = class_weights / tf.reduce_sum(class_weights)
    CLASSES_NUMBER = 6
    alpha = tf.reshape(class_weights,[CLASSES_NUMBER,1])
    print((alpha))
    gamma = 2.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.math.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.matmul(tf.multiply(weight, ce), alpha)
    loss = tf.reduce_mean(fl)
    return loss


def dice_loss(y_true, y_pred,smooth = 1):
    print(y_true)
    class_count = tf.reduce_sum(y_true,[0,1,2]) + 1
    class_weights = tf.reduce_sum(class_count) /  class_count
    
    w, m1, m2 = class_weights , y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * tf.reduce_sum(w * intersection) + smooth) / \
    (tf.reduce_sum(w * m1) + tf.reduce_sum(w * m2) + smooth)
    loss_value = 1. - tf.reduce_sum(score)
    return loss_value

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
    "ball_iou"       : 0,
    "field_iou"      : 1,
    "robots_iou"     : 2,
    "line_iou"       : 3,
    "background_iou" : 4,
    "goal_iou"       : 5,
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
        return self.intersection/(self.union + 0.0000000001)
class total_mean_iou(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


def class_weights(m):
    def update_state(y_true, y_pred):
        class_count = tf.reduce_sum(y_true,[0,1,2]) + 1
        class_weights = tf.reduce_sum(class_count) /  class_count
        return class_weights[m]
    update_state.__name__ = 'class_weights' + str(m)
    return update_state


# class class_weights(tf.keras.metrics.Metric):
#     def __init__(self, name='class_weights', **kwargs):
#         super(class_weights, self).__init__(name=name, **kwargs)
#         # self.normalized_weights = []
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         print("cccasdddddddddddddddddddddddddd",y_true)
#         weights = tf.reduce_sum(y_true,[0,1,2])
#         self.normalized_weights = weights / tf.reduce_sum(weights) 
#         # self.normalized_weights = tf.shape(y_true)
#     def result(self):
#         return tf.gather(self.normalized_weights, objects_id[self.name])