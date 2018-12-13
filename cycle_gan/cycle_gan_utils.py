import scipy
import keras
# from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
import keras_contrib
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
import tensorflow as tf

def build_generator(img_shape, channels, gf):

    def conv2d(layer_input, filters, f_size =4):
        d = Conv2D(filters, kernel_size=f_size, strides = 2, padding = 'same', data_format= 'channels_last')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = BatchNormalization()(d)
        return d


    def deconv2d(layer_input, skip_input, filters, f_size = 4, dropout_rate =0):
        u = UpSampling2D(size = 2, data_format='channels_last')(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides = 1, padding = 'same', activation= 'relu', data_format = 'channels_last')(u)

        if dropout_rate:
            u = Dropout(dropout_rate)(u)

        u = BatchNormalization()(u)
        u = Concatenate()([u, skip_input])

        return u


    # downsampling
    d0 = Input(shape = img_shape)
    d1 = conv2d(d0, gf)
    print(type(d1))

    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)

    # upsampling
    u1 = deconv2d(d4,d3, gf*4)
    u2 = deconv2d(u1,d2, gf*2)
    u3 = deconv2d(u2,d1, gf)
    u4 = UpSampling2D(size = 2, data_format='channels_last')(u3)

    output_img = Conv2D(channels, kernel_size=4, strides = 1, padding = 'same', activation = 'tanh', data_format= 'channels_last')(u4)

    return Model(d0, output_img)


def build_discriminator(img_shape, df):

    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same', data_format = 'channels_last')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = BatchNormalization()(d)
        return d

    img = Input(shape = img_shape)
    d1 = d_layer(img, df, normalization=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same', data_format='channels_last')(d4)

    return Model(img, validity)



def build_encoder(input_dim, output_dim, penalty = 1.0e-3):

    kwargs = {
        'kernel_regularizer': l2_regularizer(penalty),
        'bias_regularizer': l2_regularizer(penalty)}

    h = Input(shape = (input_dim,))
    output = Dense(output_dim, activation = 'relu', name = 'code', **kwargs)(h)

    return Model(h, output)



def build_classifier(input_dim, hidden_units, penalty = 1.0e-3):


    kwargs = {
    'kernel_regularizer': l2_regularizer(penalty),
    'bias_regularizer': l2_regularizer(penalty)}
    input = Input(shape = (input_dim,))


    for i,hidden_unit in enumerate(hidden_units):

        if i == 0:
            h = Dense(hidden_unit, activation = 'relu', **kwargs)(input)
        else:
            h = Dense(hidden_unit, activation = 'relu', **kwargs)(h)

    logits = Dense(1, activation = 'sigmoid', **kwargs)(h)

    return Model(input , logits)





def discriminator_fn(code, code2, hidden_units):

    kwargs = {
    'kernel_regularizer': l2_regularizer(1.0e-3),
    'bias_regularizer': l2_regularizer(1.0e-3)}
    h = code

    for hidden_unit in hidden_units:
        h = Dense(hidden_unit, activation=tf.nn.leaky_relu, **kwargs)(h)
        h = Dropout(rate = 0.3)(h)

    h = Dense(1, name = 'dis_out', **kwargs)(h)
    logits = h

    return logits