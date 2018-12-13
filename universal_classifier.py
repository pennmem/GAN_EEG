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
from sklearn.metrics import roc_auc_score

from cycle_gan.load_data import *
from cycle_gan.cycle_gan_utils import *

rhino_root = '/Volumes/RHINO'
subjects = os.listdir(rhino_root + '/scratch/tphan/joint_classifier/FR1')



optimizer = Adam(0.002,0.5)
output_dim = 10 # tunable

code_input = Input(shape = (output_dim,))
discriminator = build_classifier(output_dim, hidden_units = [50,25])

guess = discriminator(code_input)
model_discriminator = Model(code_input, guess)
model_discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])



discriminator.trainable = False

model_encoder_static_vec = []


subjects_vec = []
train_data = []
test_data = []
sample_weights_vec = []


#subjects_test = ['R1001P', 'R1002P', 'R1003P', 'R1010J', 'R1390M', 'R1031M']

subject_exclude = 'R1065J'


for i,subject in enumerate(subjects):

    if subject!= subject_exclude:
        print(i)
        try:

            if i == (len(subjects)-1):
                (train_x, train_y), (test_x, test_y) = load_data(subject, 'FR1', session = 0)
            else:
                (train_x, train_y), (test_x, test_y) = load_data(subject, 'FR1')


            sample_weights,_,_ = get_sample_weights_fr(train_y)

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
            test_data.append([test_x,test_y])
            sample_weights_vec.append(sample_weights)
        except:
            print(subject)


batch_size = 12

for i in np.arange(200):

    for j in np.arange(len(subjects_vec)-1):
        try:

            idx = np.random.choice(np.arange(train_data[j][0].shape[0]),batch_size, replace = False)
            train_x_batch, train_y_batch = train_data[j][0].values[idx], train_data[j][1].values[idx]
            batch_weight = sample_weights_vec[j][idx]

            loss1 = model_encoder_static_vec[j].train_on_batch(train_x_batch, train_y_batch, sample_weight= batch_weight)
            model_j = model_encoder_static_vec[j]
            intermediate_layer_model = Model(inputs = model_j.layers[1].inputs, outputs = model_j.layers[1].outputs)
            code_x_batch = intermediate_layer_model.predict(train_x_batch)


            weights_before = model_discriminator.layers[1].get_weights()
            loss2 = model_discriminator.train_on_batch(code_x_batch, train_y_batch, sample_weight= batch_weight)
            weights_after = model_discriminator.layers[1].get_weights()


            if i%20 == 0:
                print((loss1,loss2))
        except:
            print(j)



# apply to last subject


subject = subject_exclude
subject_dir = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/dataset.pkl'
dataset = joblib.load(subject_dir)
dataset_enc = select_phase(dataset)
dataset_enc['X'] = normalize_sessions(dataset_enc['X'], dataset_enc['session'])   # select only encoding data

sessions = np.unique(dataset_enc['session'])


y_vec = []
prob_vec = []
for session in sessions:

    (train_x, train_y), (test_x, test_y) = load_data(subject, 'FR1', session = session)

    sample_weights,_,_ = get_sample_weights_fr(train_y)

    input_dim = train_x.shape[1]
    encoder = build_encoder(input_dim, output_dim)
    eeg_input = Input(shape = (input_dim,))
    code = encoder(eeg_input)
    guess_static = discriminator(code)

    # static encoder
    model_encoder = Model(inputs = eeg_input, outputs = guess_static)
    model_encoder.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])

    model_encoder_static_vec[j] = model_encoder

    for i in np.arange(200):
        idx = np.random.choice(np.arange(train_x.shape[0]),batch_size, replace = False)
        train_x_batch, train_y_batch = train_x.values[idx], train_y.values[idx]
        batch_weight = sample_weights[idx]

        loss1 = model_encoder.train_on_batch(train_x_batch, train_y_batch, sample_weight= batch_weight)
        intermediate_layer_model = Model(inputs = model_encoder.layers[1].inputs, outputs = model_encoder.layers[1].outputs)
        code_x_batch = intermediate_layer_model.predict(train_x_batch)

        if i%10==0:
            print(loss1)


    probs = model_encoder.predict(test_x)
    y_vec.append(test_y)
    prob_vec.append(probs)


        # weights_before = model_discriminator.layers[1].get_weights()
        # loss2 = model_discriminator.train_on_batch(code_x_batch, train_y_batch, sample_weight= batch_weight)
        # weights_after = model_discriminator.layers[1].get_weights()

y_vec = np.concatenate(y_vec)
prob_vec = np.concatenate(prob_vec)
auc = roc_auc_score(y_vec, prob_vec)
print(auc)