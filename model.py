
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def conv_block(inputs, filters, pooling=True):
    x = layers.Conv2D(filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if pooling == True:
        pooled = layers.MaxPool2D((2, 2))(x)
        return x, pooled
    else:
        return x

def unet_model(input_shape, num_classes):
    inputs = layers.Input(input_shape)
    x0 = layers.experimental.preprocessing.Rescaling(1./255, offset=0.0)(inputs)
    x1, p1 = conv_block(x0, 16, pooling=True)
    x2, p2 = conv_block(p1, 32, pooling=True)
    x3, p3 = conv_block(p2, 48, pooling=True)
    x4, p4 = conv_block(p3, 64, pooling=True)
    b1 = conv_block(p4, 128, pooling=False)
    u1 = layers.UpSampling2D((2, 2), interpolation="bilinear")(b1)
    c1 = layers.Concatenate()([u1, x4])
    x5 = conv_block(c1, 64, pooling=False)
    u2 = layers.UpSampling2D((2, 2), interpolation="bilinear")(x5)
    c2 = layers.Concatenate()([u2, x3])
    x6 = conv_block(c2, 48, pooling=False)
    u3 = layers.UpSampling2D((2, 2), interpolation="bilinear")(x6)
    c3 = layers.Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, pooling=False)
    u4 = layers.UpSampling2D((2, 2), interpolation="bilinear")(x7)
    c4 = layers.Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, pooling=False)
    output = layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(x8)
    return Model(inputs, output)
