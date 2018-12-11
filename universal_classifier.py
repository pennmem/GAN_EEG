import scipy
import keras
# from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import keras
from keras.engine.topology import Network
from keras.models import load_model
from load_eeg import*
from keras.engine.topology import Network

from cycle_gan.load_data import *
from cycle_gan.cycle_gan_utils import *

rhino_root = '/Volumes/RHINO'
subjects = os.listdir(rhino_root + '/scratch/tphan/joint_classifier/FR1')



optimizer = Adam(0.002,0.5)
output_dim = 100 # tunable

code_input = Input(shape = (output_dim,))
discriminator = build_classifier(output_dim, hidden_units = [50,25])

guess = discriminator(code_input)
model_discriminator = Model(code_input, guess)
model_discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])



discriminator.trainable = False

model_encoder_static_vec = []


subjects_vec = []
train_data = []
for subject in subjects[:10]:

    try:
        (train_x, train_y), (test_x, test_y) = load_data(subject, 'FR1')

        input_dim = train_x.shape[1]
        encoder = build_encoder(input_dim, output_dim)
        eeg_input = Input(shape = (input_dim,))

        code = encoder(eeg_input)
        guess_static = discriminator(code)

        # static encoder
        model_encoder = Model(inputs = eeg_input, outputs = guess_static)
        model_encoder.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])

        model_encoder_static_vec.append(model_encoder)
        subjects_vec.append(subject)
        train_data.append([train_x,train_y])
    except:
        print(subject)



batch_size = 32

for i in np.arange(100):


    for j in np.arange(len(subjects_vec)):

        idx = np.random.choice(np.arange(train_data[j][0].shape[0]),32, replace = True)
        train_x_batch, train_y_batch = train_data[j][0].values[idx], train_data[j][1].values[idx]

        loss1 = model_encoder_static_vec[j].train_on_batch(train_x_batch, train_y_batch)

        code_x_batch = model_encoder_static_vec[j].predict(train_x_batch)
        loss2 = model_discriminator.train_on_batch(train_x_batch, train_y_batch)

        print(loss2)


    # if i%100 ==0:
    #     eval = model_encoder_static.evaluate(test_x, test_y)
    #     print("[loss: {:.5}, acc: {:.5}%".format(eval[0], eval[1]))

