# Import Libraries
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from keras import backend as K
import model
import datetime

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 80% train 15% validation 5% test
data_train = tfds.load("humanoidSoccerDataset", split='train[:80%]', shuffle_files=True)
data_valid  = tfds.load("humanoidSoccerDataset", split='train[80%:]', shuffle_files=True)
data_test =tfds.load("humanoidSoccerDataset", split='test', shuffle_files=True)

# 2. Show some dataset
train_to_show = data_train.map(
    lambda data: (tf.cast(data["image"], tf.float32)/255., data['label'])
).batch(
    64
)
valid_to_show = data_valid.map(
    lambda data: (tf.cast(data["image"], tf.float32)/255., data['label'])
).batch(
    64
)
train_to_calculate_weights = iter(train_to_show)

t_iterator = iter(train_to_show)
v_iterator = iter(valid_to_show)
t_next_val = t_iterator.get_next()
v_next_val = v_iterator.get_next()

# plt.figure(figsize = (12,18), dpi = 300)
t_buf = t_next_val
v_buf = v_next_val

for ii in range(2):
    print(t_buf[0][ii])
    plt.subplot(2,4,ii*2+1)
    plt.imshow(t_buf[0][ii])
    plt.axis("off")
    plt.title("Image Train")
    plt.subplot(2,4,ii*2+2)
    plt.imshow(t_buf[1][ii])
    plt.axis("off")
    plt.title("Image Train")
for ii in range(2,4):
    plt.subplot(2,4,ii*2+1)
    plt.imshow(v_buf[0][ii])
    plt.axis("off")
    plt.title("Image Valid")
    plt.subplot(2,4,ii*2+2)
    plt.imshow(v_buf[1][ii])
    plt.axis("off")
    plt.title("Label Valid")
plt.show()

# 3. One-hot encoding
CLASSES_NUMBER = 6
BATCH_SIZE = 40


palette = np.array([
    [31.,120.,180. ] , # Ball
    [106.,176.,25. ] , # Field
    [156.,62.,235. ] , # Robots
    [255.,255.,255.] , # Line
    [69.,144.,232. ] , # Background
    [227.,26.,28.  ] , # Goal
    ], dtype=np.float32)

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
# 3. Data normalization and Augmentation
def augmentor(data_set):
    # c_type = tf.float32
    ds = data_set.map(
        lambda data: (tf.image.convert_image_dtype(data["image"], tf.float32), _one_hot_encode(data["label"]))
    ).map(
        lambda image, label: (tf.image.random_hue(image, 0.12), label)
    ).map(
        lambda image, label: (tf.image.random_saturation(image, 1, 3), label)
    ).map(
        lambda image, label: (random_flip_example(image, label))
    ).map(
        lambda image, label: (tf.add(image, tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=.02)), label)
    ).batch(
        BATCH_SIZE
    ).repeat()

    return ds

train_data = augmentor(data_train)
# iterat = iter(train_data)
# for it in iterat:
#     print(np.min(it[0]),np.max(it[0]))
valid_data = augmentor(data_valid)


# 4. Show some dataset after augmentation
objects = [
    "Ball",
    "Field",
    "Robots",
    "Line",
    "Background",
    "Goals"
]
t_iterator2 = iter(train_data)
v_iterator2 = iter(valid_data)
t_next_val2 = t_iterator2.get_next()
v_next_val2 = v_iterator2.get_next()

t_buf2 = t_next_val2
v_buf2 = v_next_val2

plt.figure(figsize = (12,18), dpi = 300)

plt.subplot(2,7,1)
plt.imshow(t_buf2[0][0])
plt.axis("off")
plt.title("Train Aug")

for ii in range(6):
    plt.subplot(2,7,ii+2)
    plt.imshow(t_buf2[1][0][:,:,ii], cmap='gray')

plt.subplot(2,7,8)
plt.imshow(v_buf2[0][0])
plt.axis("off")
plt.title("Valid Aug")

for ii in range(6):
    plt.subplot(2,7,ii+9)
    plt.imshow(v_buf2[1][0][:,:,ii], cmap='gray')
plt.show()

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

model = model.unet_model((240, 320, 3), 6)
# print(model.summary())
# lr = 1
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
        self.intersection = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_and(true_class_map,pred_class_map)),tf.float32)
        self.union = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_or(true_class_map,pred_class_map)),tf.float32)
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
        self.intersection = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_and(true_class_map,pred_class_map)),tf.float32)
        self.union = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_or(true_class_map,pred_class_map)),tf.float32)
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
        self.intersection = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_and(true_class_map,pred_class_map)),tf.float32)
        self.union = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_or(true_class_map,pred_class_map)),tf.float32)
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
        self.intersection = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_and(true_class_map,pred_class_map)),tf.float32)
        self.union = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_or(true_class_map,pred_class_map)),tf.float32)
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
        self.intersection = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_and(true_class_map,pred_class_map)),tf.float32)
        self.union = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_or(true_class_map,pred_class_map)),tf.float32)
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
        self.intersection = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_and(true_class_map,pred_class_map)),tf.float32)
        self.union = tf.cast(tf.reduce_sum(tf.bitwise.bitwise_or(true_class_map,pred_class_map)),tf.float32)
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

# 5. Defining Callbacks
model.save_weights('/home/mrl/Desktop/model/FINAL-MODEL/Humanoid.h5')
model_name = "Humanoid.h5"
log_dir = "Logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 30)

monitor = tf.keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss',\
                                             verbose=0,save_best_only=True,\
                                             save_weights_only=True,\
                                             mode='min')
# Learning rate schedule
def scheduler(epoch, lr):
    if epoch%20 == 0 and epoch!= 0:
        lr = lr/2
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)


# 6. Train the model
EPOCHS = 420
STEPS_PER_EPOCH = 250
VALIDATION_STEPS = 5
callbacks = [early_stop, monitor, lr_schedule, tensorboard_callback]

model_history = model.fit(train_data, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=valid_data,
                          callbacks=callbacks)

print(model_history.history.keys())
# plot loss and validation loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Weigted Dice Loss','Validation Weigted Dice Loss'], loc='upper right')
plt.savefig('loss-plot.svg')
plt.show()
# plot iou per class for training
plt.plot(model_history.history['ball_mean_iou'])
plt.plot(model_history.history['field_mean_iou'])
plt.plot(model_history.history['robot_mean_iou'])
plt.plot(model_history.history['line_mean_iou'])
plt.plot(model_history.history['backGND_mean_iou'])
plt.plot(model_history.history['goal_mean_iou'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Ball Training Iou','Field Training Iou', 'Robot Training Iou', 'Lines Training Iou', 'Background Training Iou', 'Goal Training Iou'], loc='upper left')
plt.savefig('iou-train-per-class-plot.svg')
plt.show()
# plot iou per class for validation
plt.plot(model_history.history['val_ball_mean_iou'])
plt.plot(model_history.history['val_field_mean_iou'])
plt.plot(model_history.history['val_robot_mean_iou'])
plt.plot(model_history.history['val_line_mean_iou'])
plt.plot(model_history.history['val_backGND_mean_iou'])
plt.plot(model_history.history['val_goal_mean_iou'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Ball Validation IoU','Field Validation IoU', 'Robot Validation IoU', 'Lines Validation IoU', 'Background Validation IoU', 'Goal Validation IoU'], loc='upper left')
plt.savefig('iou-valid-per-class-plot.svg')
plt.show()
# plot dice coef jacard loss and mean iou training
plt.plot(model_history.history['dice_coef'])
plt.plot(model_history.history['jaccard_coef'])
plt.plot(model_history.history['my_mean_iou'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training Dice Coefficient','Training Jaccard Coefficient', 'Training Mean IoU'], loc='upper left')
plt.savefig('djm-TRAIN.svg')
plt.show()
# plot dice coef jacard loss and mean iou Validation
plt.plot(model_history.history['val_dice_coef'])
plt.plot(model_history.history['val_jaccard_coef'])
plt.plot(model_history.history['val_my_mean_iou'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Validation Dice Coefficient','Validation Jaccard Coefficient', 'Validation Mean IoU'], loc='upper left')
plt.savefig('djm-VALID.svg')
plt.show()
#plot val and train 
plt.plot(model_history.history['dice_coef'])
plt.plot(model_history.history['jaccard_coef'])
plt.plot(model_history.history['my_mean_iou'])
plt.plot(model_history.history['val_dice_coef'])
plt.plot(model_history.history['val_jaccard_coef'])
plt.plot(model_history.history['val_my_mean_iou'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training Dice Coefficient','Training Jaccard Coefficient', 'Training Mean IoU','Validation Dice Coefficient','Validation Jaccard Coefficient', 'Validation Mean IoU'], loc='upper left')
plt.savefig('djm.svg')
plt.show()