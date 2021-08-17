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
    "Ball"      :np.array([[255.  , 0.  , 0.]],dtype=np.float32),
    "Field"     :np.array([[0.  , 255., 0.  ]],dtype=np.float32),
    "Line"      :np.array([[255., 255., 255.],[255., 0.  , 255.]],dtype=np.float32),
    "Background":np.array([[0.  , 0.  , 0.  ],[0.  , 0.  , 255.],[127, 127, 127]],dtype=np.float32),
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

t_buf = t_next_val
v_buf = v_next_val

for ii in range(2):
    print(t_buf[0][ii])
    plt.subplot(2, 4, ii*2+1)
    plt.imshow(t_buf[0][ii])
    plt.axis("off")
    plt.title("Image Train")
    plt.subplot(2, 4, ii*2+2)
    plt.imshow(t_buf[1][ii])
    plt.axis("off")
    plt.title("Image Train")
for ii in range(2, 4):
    plt.subplot(2, 4, ii*2+1)
    plt.imshow(v_buf[0][ii])
    plt.axis("off")
    plt.title("Image Valid")
    plt.subplot(2, 4, ii*2+2)
    plt.imshow(v_buf[1][ii])
    plt.axis("off")
    plt.title("Label Valid")
plt.show()

# 3. One-hot encoding


# 3. Data normalization and Augmentation


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

total_iou      = losses.total_mean_iou(6)
ball_iou       = losses.object_mean_iou("ball_iou")
field_iou      = losses.object_mean_iou("field_iou")
robots_iou     = losses.object_mean_iou("robot_iou")
line_iou       = losses.object_mean_iou("line_iou")
background_iou = losses.object_mean_iou("back_gnd_iou")
goal_iou       = losses.object_mean_iou("goal_iou")

metrics =[losses.dice_coef, losses.jaccard_coef, ball_iou, field_iou, robots_iou, line_iou, background_iou, goal_iou, total_iou]
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