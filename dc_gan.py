# utility for gans
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Dropout, BatchNormalization, \
    LeakyReLU, Activation, UpSampling2D, Conv2D, MaxPooling2D,  Flatten
from tensorflow.keras import Sequential, Input, Model
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
from tensorflow.contrib.gan.python.namedtuples import GANTrainSteps
from functools import partial



def generator_model():
    model = Sequential()
    model.add(Dense(1024, input_dim=100))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(LeakyReLU(0.2))
    return model




def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
            )
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
