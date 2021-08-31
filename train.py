"""Import Libraries"""
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
data_train  = tfds.load("humanoidSoccerDataset", split='train[:80%]', shuffle_files=True)
data_valid  = tfds.load("humanoidSoccerDataset", split='train[80%:]', shuffle_files=True)
# data_test   = tfds.load("humanoidSoccerDataset", split='test', shuffle_files=True)

train_size = sum(1 for _ in data_train)
valid_size = sum(1 for _ in data_valid)

img_width = 320
img_height = 240
BATCH_SIZE = 12

palette = {
    "Ball"       :np.array([[180., 120.,  31.]],dtype=np.float32),
    "Field"      :np.array([[25. , 176., 106.]],dtype=np.float32),
    "Robots"     :np.array([[235.,  62., 156.]],dtype=np.float32),
    "Line"       :np.array([[255., 255., 255.]],dtype=np.float32),
    "Background" :np.array([[232., 144.,  69.]],dtype=np.float32),
    "Goal"       :np.array([[28. ,  26., 227.]],dtype=np.float32),
}
CLASSES_NUMBER = len(palette.keys())

def _one_hot_encode(img):
    """Converts mask to a one-hot encoding specified by the semantic map."""
    semantic_map = []
    for category in palette :
        class_map = tf.zeros((img_height,img_width), dtype=tf.dtypes.bool, name=None)
        for color in palette[category]:
            p_map = tf.reduce_all(tf.equal(img, color), axis=-1)
            class_map = tf.math.logical_or(p_map,class_map)
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
        BATCH_SIZE
    ).repeat()
    return ds
# 2. Show some dataset
# train_to_show = data_train.map(
#     lambda data: (tf.cast(data["image"], tf.float32)/255., data['label'])
# ).batch(
#     64
# )
# valid_to_show = data_valid.map(
#     lambda data: (tf.cast(data["image"], tf.float32)/255., data['label'])
# ).batch(
#     64
# )
# train_to_calculate_weights = iter(train_to_show)

# t_iterator = iter(train_to_show)
# v_iterator = iter(valid_to_show)
# t_next_val = t_iterator.get_next()
# v_next_val = v_iterator.get_next()

# t_buf = t_next_val
# v_buf = v_next_val

# for ii in range(2):
#     # print(t_buf[0][ii])
#     plt.subplot(2, 4, ii*2+1)
#     plt.imshow(t_buf[0][ii])
#     plt.axis("off")
#     plt.title("Train Image")
#     plt.subplot(2, 4, ii*2+2)
#     plt.imshow(t_buf[1][ii])
#     plt.axis("off")
#     plt.title("Train Label")
# for ii in range(2, 4):
#     plt.subplot(2, 4, ii*2+1)
#     plt.imshow(v_buf[0][ii])
#     plt.axis("off")
#     plt.title("Valid Image")
#     plt.subplot(2, 4, ii*2+2)
#     plt.imshow(v_buf[1][ii])
#     plt.axis("off")
#     plt.title("Valid Label")
# plt.show()

# 3. One-hot encoding



train_data = augmentor(data_train)
valid_data = augmentor(data_valid)
print("asdfasdasdf")

plot_dataset = False

if plot_dataset:
    def display_sample(display_list):
        """Show side-by-side an input image,
        the ground truth and the prediction.
        """
        # print(tf.math.reduce_max (image),tf.math.reduce_min (image))
        plt.figure(figsize=(18, 18))

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()

    for image, mask in train_data.take(100):
        display_sample([image[0], mask[0]])


# 3. Data normalization and Augmentation

# 4. Show some dataset after augmentation /////////////////////
objects = [
    "Ball"      ,
    "Field"     ,
    "Robots"    ,
    "Line"      ,
    "Background",
    "Goal"      ,
]
# t_iterator = iter(train_data)
# v_iterator = iter(valid_data)
# t_next_val = t_iterator.get_next()
# v_next_val = v_iterator.get_next()

# plt.figure(figsize = (12,18), dpi = 300)

# plt.subplot(2,7,1)
# plt.imshow(t_next_val[0][0]/255.)
# plt.axis("off")
# plt.title("Train Aug")

# for ii in range(6):
#     plt.subplot(2,7,ii+2)
#     plt.imshow(t_next_val[1][0][:,:,ii], cmap='gray')

# plt.subplot(2,7,8)
# plt.imshow(v_next_val[0][0]/255.)
# plt.axis("off")
# plt.title("Valid Aug")

# for ii in range(6):
#     plt.subplot(2,7,ii+9)
#     plt.imshow(v_next_val[1][0][:,:,ii], cmap='gray')
# plt.show()

# ////////////////////////////////////////////////////////////

model = model.unet_model((img_height, img_width, 3), CLASSES_NUMBER)

total_iou      = losses.total_mean_iou(CLASSES_NUMBER)
ball_iou       = losses.object_mean_iou("ball_iou"       )
field_iou      = losses.object_mean_iou("field_iou"      )
robots_iou     = losses.object_mean_iou("robots_iou"     )
line_iou       = losses.object_mean_iou("line_iou"       )
background_iou = losses.object_mean_iou("background_iou" )
goal_iou       = losses.object_mean_iou("goal_iou"       )
cls_w = [losses.class_weights(m) for m in range(5)]

metrics =[losses.dice_coef, losses.jaccard_coef, ball_iou, field_iou, line_iou, background_iou, total_iou]

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=losses.multi_category_focal_loss1, metrics=metrics)

# 5. Defining Callbacks
# model.save_weights('/home/mrl/Desktop/model/FINAL-MODEL/Humanoid.h5')
model_name = "/home/mrl/semantic segmentation article/models/Humanoid.h5"
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


# # 6. Train the model
EPOCHS = 420
STEPS_PER_EPOCH = 666
VALIDATION_STEPS = 5
callbacks = [early_stop, monitor, lr_schedule, tensorboard_callback]

model_history = model.fit(train_data, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=valid_data,
                          callbacks=callbacks)


#TODO
"""
model_name = "models/segmentation"
log_dir = "Logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 30)

monitor = tf.keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss',\
                                             verbose=0,save_best_only=True,\
                                             save_weights_only=False,\
                                             mode='min')
# Learning rate schedule
def scheduler(epoch, lr):
    if epoch%20 == 0 and epoch!= 0:
        lr = lr/2
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)


# 6. Train the model
EPOCHS = 420
STEPS_PER_EPOCH = train_size / BATCH_SIZE
VALIDATION_STEPS = valid_size / BATCH_SIZE
callbacks = [early_stop, monitor, lr_schedule, tensorboard_callback]

model_history = model.fit(train_data, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=valid_data,
                          callbacks=callbacks) 
"""