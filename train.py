# Import Libraries
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from keras import backend as K
import model
import datetime
import losses

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
t_iterator = iter(train_data)
v_iterator = iter(valid_data)
t_next_val = t_iterator.get_next()
v_next_val = v_iterator.get_next()

plt.figure(figsize = (12,18), dpi = 300)

plt.subplot(2,7,1)
plt.imshow(t_next_val[0][0])
plt.axis("off")
plt.title("Train Aug")

for ii in range(6):
    plt.subplot(2,7,ii+2)
    plt.imshow(t_next_val[1][0][:,:,ii], cmap='gray')

plt.subplot(2,7,8)
plt.imshow(v_next_val[0][0])
plt.axis("off")
plt.title("Valid Aug")

for ii in range(6):
    plt.subplot(2,7,ii+9)
    plt.imshow(v_next_val[1][0][:,:,ii], cmap='gray')
plt.show()

model = model.unet_model((240, 320, 3), 6)

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

metrics =[losses.dice_coef, losses.jaccard_coef,ball_iou,field_iou,robots_iou,line_iou,background_iou,goal_iou,myiou]
object_weights = np.array([1-0.008129217, 1-0.741332343, 1-0.038759669, 1-0.033971285, 1-0.159327414, 1-0.018480072])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=losses.weighted_dice_loss(weights=object_weights), metrics=metrics)

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