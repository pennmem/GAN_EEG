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


def build_generator(img_shape, channels, n_filters = 4):

    def conv2d(layer_input, filters, f_size =4):
        d = Conv2D(filters, kernel_size=f_size, strides = 2, padding = 'same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d


    def deconv2d(layer_input, skip_input, filters, f_size = 4, dropout_rate =0):
        u = UpSampling2D(size = 2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides = 1, padding = 'same', activation= 'relu')(u)

        if dropout_rate:
            u = Dropout(dropout_rate)(u)

        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])

        return u


    # downsampling
    d0 = Input(shape = img_shape)
    d1 = conv2d(d0, n_filters)
    print(type(d1))

    d2 = conv2d(d1, n_filters*2)
    d3 = conv2d(d2, n_filters*4)
    d4 = conv2d(d3, n_filters*8)

    # upsampling
    u1 = deconv2d(d4,d3,n_filters*4)
    u2 = deconv2d(u1,d2,n_filters*2)
    u3 = deconv2d(u2,d1, n_filters)
    u4 = UpSampling2D(size = 2)(u3)

    output_img = Conv2D(channels, kernel_size=4, strides = 1, padding = 'same', activation = 'tanh')(u4)

    return Model(d0, output_img)


def build_discriminator(img_shape, n_filters):

    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    img = Input(shape = img_shape)
    d1 = d_layer(img, n_filters, normalization=False)
    d2 = d_layer(d1, n_filters*2)
    d3 = d_layer(d2, n_filters*4)
    d4 = d_layer(d3, n_filters*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model(img, validity)

