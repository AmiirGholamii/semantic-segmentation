import timeit
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import backend as K
import cv2
import numpy as np
import model
import os
import humanoid_soccer_dataset
import random
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

CLASSES_NUMBER = 6

palette = np.array([
    [  0.],
    [  1.],
    [  2.],
    [  3.],
    [  4.],
    [  5.],
    ], dtype=np.float32)
palette = np.array([
    [31.,120.,180. ] , # Ball
    [106.,176.,25. ] , # Field
    [156.,62.,235. ] , # Robots
    [255.,255.,255.] , # Line
    [69.,144.,232. ] , # Background
    [227.,26.,28.  ] , # Goal
], dtype=np.float32)

model_name = "/home/mrl/semantic-segmentation-article/models/Humanoid.h5"
# model_name = "/home/mrl/Desktop/model/trl/trl.h5"

model = model.unet_model((240, 320, 3), 6)
model.load_weights(model_name)

# # TODO
ALPHA = 0.8
GAMMA = 2

def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
    
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



def weighted_dice_loss(y_true, y_pred):
    weight = np.array([1-0.008129217,1-0.741332343,1-0.038759669,1-0.033971285,1-0.159327414,1-0.018480072])
    # name="MRL_DICE_LOSS"
    """
    :param y_true:
    :param y_pred:
    :param weight:
    :param name:
    :return:
    """
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
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
data_test =tfds.load("humanoidSoccerDataset", split='test', shuffle_files=True)
def tf_count(t, val):
    as_ints = tf.cast(t, tf.int32)
    tf.print(as_ints)
    count = tf.reduce_sum(as_ints)
    return count
class BallMeanIOU(tf.keras.metrics.Metric):
    def __init__(self, name='ball_mean_iou', **kwargs):
        super(BallMeanIOU, self).__init__(name=name, **kwargs)
        self.intersection = 0.0
        self.union = 0.0
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_1D_map = tf.argmax(y_true, axis=-1)
        y_pred_1D_map = tf.argmax(y_pred, axis=-1)
        true_class_map = tf.cast(tf.equal(y_true_1D_map, 0),dtype=tf.int32)
        pred_class_map = tf.cast(tf.equal(y_pred_1D_map, 0),dtype=tf.int32)
        self.intersection = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_and(true_class_map,pred_class_map)),tf.float32)+1e-4
        self.union = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_or(true_class_map,pred_class_map)),tf.float32)+1e-4
    def result(self):
        return self.intersection/self.union

class FieldMeanIOU(tf.keras.metrics.Metric):
    def __init__(self, name='field_mean_iou', **kwargs):
        super(FieldMeanIOU, self).__init__(name=name, **kwargs)
        self.intersection = 0.0
        self.union = 0.0
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_1D_map = tf.argmax(y_true, axis=-1)
        y_pred_1D_map = tf.argmax(y_pred, axis=-1)
        true_class_map = tf.cast(tf.equal(y_true_1D_map, 1),dtype=tf.int32)
        pred_class_map = tf.cast(tf.equal(y_pred_1D_map, 1),dtype=tf.int32)
        self.intersection = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_and(true_class_map,pred_class_map)),tf.float32)+1e-4
        self.union = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_or(true_class_map,pred_class_map)),tf.float32)+1e-4
    def result(self):
        return self.intersection/self.union
class RobotMeanIOU(tf.keras.metrics.Metric):
    def __init__(self, name='robot_mean_iou', **kwargs):
        super(RobotMeanIOU, self).__init__(name=name, **kwargs)
        self.intersection = 0.0
        self.union = 0.0
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_1D_map = tf.argmax(y_true, axis=-1)
        y_pred_1D_map = tf.argmax(y_pred, axis=-1)
        true_class_map = tf.cast(tf.equal(y_true_1D_map, 2),dtype=tf.int32)
        pred_class_map = tf.cast(tf.equal(y_pred_1D_map, 2),dtype=tf.int32)
        self.intersection = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_and(true_class_map,pred_class_map)),tf.float32)+1e-4
        self.union = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_or(true_class_map,pred_class_map)),tf.float32)+1e-4
        # tf.print("\nintersection: ",self.intersection)
        # tf.print("\nunion: ",self.union)
    def result(self):
        return self.intersection/self.union
class LineMeanIOU(tf.keras.metrics.Metric):
    def __init__(self, name='line_mean_iou', **kwargs):
        super(LineMeanIOU, self).__init__(name=name, **kwargs)
        self.intersection = 0.0
        self.union = 0.0
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_1D_map = tf.argmax(y_true, axis=-1)
        y_pred_1D_map = tf.argmax(y_pred, axis=-1)
        true_class_map = tf.cast(tf.equal(y_true_1D_map, 3),dtype=tf.int32)
        pred_class_map = tf.cast(tf.equal(y_pred_1D_map, 3),dtype=tf.int32)
        self.intersection = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_and(true_class_map,pred_class_map)),tf.float32)+1e-4
        self.union = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_or(true_class_map,pred_class_map)),tf.float32)+1e-4
    def result(self):
        return self.intersection/self.union
class BackgroundMeanIOU(tf.keras.metrics.Metric):
    def __init__(self, name='backGND_mean_iou', **kwargs):
        super(BackgroundMeanIOU, self).__init__(name=name, **kwargs)
        self.intersection = 0.0
        self.union = 0.0
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_1D_map = tf.argmax(y_true, axis=-1)
        y_pred_1D_map = tf.argmax(y_pred, axis=-1)
        true_class_map = tf.cast(tf.equal(y_true_1D_map, 4),dtype=tf.int32)
        pred_class_map = tf.cast(tf.equal(y_pred_1D_map, 4),dtype=tf.int32)
        self.intersection = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_and(true_class_map,pred_class_map)),tf.float32)+1e-4
        self.union = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_or(true_class_map,pred_class_map)),tf.float32)+1e-4
    def result(self):
        return self.intersection/self.union
class GoalMeanIOU(tf.keras.metrics.Metric):
    def __init__(self, name='goal_mean_iou', **kwargs):
        super(GoalMeanIOU, self).__init__(name=name, **kwargs)
        self.intersection = 0.0
        self.union = 0.0
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_1D_map = tf.argmax(y_true, axis=-1)
        y_pred_1D_map = tf.argmax(y_pred, axis=-1)
        true_class_map = tf.cast(tf.equal(y_true_1D_map, 5),dtype=tf.int32)
        pred_class_map = tf.cast(tf.equal(y_pred_1D_map, 5),dtype=tf.int32)
        self.intersection = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_and(true_class_map,pred_class_map)),tf.float32)+1e-4
        self.union = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_or(true_class_map,pred_class_map)),tf.float32)+1e-4
    def result(self):
        return self.intersection/self.union
class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


myiou = MyMeanIOU(6)
ball_iou = BallMeanIOU()
field_iou = FieldMeanIOU()
robots_iou = RobotMeanIOU()
line_iou = LineMeanIOU()
background_iou = BackgroundMeanIOU()
goal_iou = GoalMeanIOU()
metrics =[dice_coef, jaccard_coef,ball_iou,field_iou,robots_iou,line_iou,background_iou,goal_iou,myiou]
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-3), loss = weighted_dice_loss, metrics = metrics)

def _one_hot_encode(img):
    """Converts mask to a one-hot encoding specified by the semantic map."""
    semantic_map = []
    for color in palette:
      class_map = tf.reduce_all(tf.equal(img, color), axis=-1)
      semantic_map.append(class_map)

    semantic_map = tf.stack(semantic_map, axis=-1)
    semantic_map = tf.cast(semantic_map, tf.float32)
    return semantic_map
def random_flip_example(image, label):
    seed = random.random()*10
    return tf.image.random_flip_left_right(image ,seed=seed),tf.image.random_flip_left_right(label ,seed=seed)
def augmentor(data_set):
    ds = data_set.map(
        lambda data: (data["image"],data["label"])
    ).map(
        lambda image, label: (tf.image.random_hue(image, 0.08), label)
    ).map(
        lambda image, label: (tf.image.random_saturation(image, 1, 3), label)
    ).map(
        lambda image, label: (random_flip_example(image, label))
    ).map(
        lambda image, label: (tf.image.random_brightness(image, 0.3), label)
    ).map(
        lambda image, label: (tf.cast(image,tf.float32) ,_one_hot_encode(label))
    ).batch(
        1
    )
    return ds
test_data = augmentor(data_test)
model.evaluate(test_data)

palette = np.array([
    [180.,120.,31. ] , # Ball
    [25.,176.,106. ] , # Field
    [235.,62.,156. ] , # Robots
    [255.,255.,255.] , # Line
    [232.,144.,69. ] , # Background
    [28.,26.,227.  ] , # Goal
    ], dtype=np.float32)
# directory = "/home/mrl/2021/best model/2/humanoid_soccer_dataset/test/image"
# directory ='/home/mrl/tensorflow_datasets/downloads/extracted/ZIP.Dataset.zip/train/image'
directory ='/home/mrl/tensorflow_datasets/downloads/extracted/ZIP.Dataset-asli.zip/Dataset-asli/train/image'
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith(".png"): 
        x_test = cv2.imread(os.path.join(directory,filename))
        x_test = cv2.resize(x_test, (320,240))
        x_test=np.expand_dims(x_test,axis=0)
        mask = np.zeros([240,320,3],dtype=np.uint8)

        start = timeit.default_timer()
        y_pred = model.predict(x_test)
        stop = timeit.default_timer()
        print('Time: ', stop - start)  
        y_pred = y_pred[0]
        for i in range(240):
            for j in range(320):
                class_num = np.argmax(y_pred[i,j])
                mask[i,j] = palette[class_num]
        cv2.imshow("asdaf",mask)
        cv2.imwrite(filename,mask)
        cv2.waitKey(0)
