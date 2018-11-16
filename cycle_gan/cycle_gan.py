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

from cycle_gan.load_data import *
from cycle_gan.cycle_gan_utils import *

imgs_rows = 256
imgs_cols = 256
channels = 3
img_shape = (imgs_rows, imgs_cols, channels)

patch = int(imgs_rows/2**4)
disc_patch = (patch, patch, 1)

gf = 32  # generator filters
df = 64   # discriminator filters

optimizer = Adam(0.0002, 0.5)


lambda_cycle = 10.0
lambda_id = 0.1*lambda_cycle

d_A = build_discriminator(img_shape, n_filters=4)
d_B = build_discriminator(img_shape, n_filters=4)

d_A.compile(loss = 'mse', optimizer = optimizer, metrics = ['accuracy'])
d_B.compile(loss = 'mse', optimizer = optimizer, metrics = ['accuracy'])


img_A = Input(shape = img_shape)
img_B = Input(shape = img_shape)
generator_A2B = build_generator(img_shape, channels)
generator_B2A = build_generator(img_shape, channels)

fake_B = generator_A2B(img_A)
fake_A = generator_B2A(img_B)

reconstrc_A = generator_B2A(fake_B)
reconstrc_B = generator_A2B(fake_A)


# identity maps
# img_A_id = generator_B2A(img_A)
# img_B_id = generator_A2B(img_B)

d_A.trainable = False
d_B.trainable = False

valid_A = d_A(fake_A)
valid_B = d_B(fake_B)

# combined = Model(inputs = [img_A, img_B], outputs = [valid_A, valid_B, reconstrc_A, reconstrc_B, img_A_id, img_B_id])
combined = Model(inputs = [img_A, img_B], outputs = [valid_A, valid_B, reconstrc_A, reconstrc_B])
combined.compile(loss = ['mse', 'mse', 'mae', 'mae'], loss_weights=[1,1,lambda_cycle, lambda_cycle],  optimizer = optimizer)
# combined.compile(loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae'], loss_weights=[1,1,lambda_cycle, lambda_cycle, lambda_id, lambda_id],  optimizer = optimizer)


n_epochs = 100
batch_size = 16

valid = np.ones((batch_size,) + disc_patch)
fake = np.zeros((batch_size,) + disc_patch)


sample_interval = 10
start_time = datetime.datetime.now()


for batch_i,epoch in enumerate(np.arange(n_epochs)):

    a_loader,b_loader = get_iphone_dataset(batch_size=batch_size)
    a_loader = iter(a_loader)
    b_loader = iter(b_loader)


    # load data
    imgs_A = a_loader.next().numpy()
    imgs_B = b_loader.next().numpy()

    imgs_A = imgs_A/127.5- 1.0
    imgs_B = imgs_B/127.5 -1.0

    fake_B = generator_A2B.predict(imgs_A)
    fake_A = generator_B2A.predict(imgs_B)

    dA_loss_real = d_A.train_on_batch(imgs_A, valid)
    dA_loss_fake = d_A.train_on_batch(fake_A, fake)
    dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

    dB_loss_real = d_B.train_on_batch(imgs_B, valid)
    dB_loss_fake = d_B.train_on_batch(fake_B, fake)
    dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

    # Total disciminator loss
    d_loss = 0.5 * np.add(dA_loss, dB_loss)

    g_loss = combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B])

    elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                            % ( epoch, n_epochs,
                                                                batch_i, batch_size,
                                                                d_loss[0], 100*d_loss[1],
                                                                g_loss[0],
                                                                np.mean(g_loss[1:3]),
                                                                np.mean(g_loss[3:5]),
                                                                np.mean(g_loss[5:6]),
                                                                elapsed_time))

    if batch_i % sample_interval == 0:
        sample_images(epoch, batch_i, generator_A2B, generator_B2A)



    r, c = 2, 3

    a_loader,b_loader = get_iphone_dataset(batch_size=1)
    a_loader = iter(a_loader)
    b_loader = iter(b_loader)

    imgs_A = a_loader.next().numpy()
    imgs_B = b_loader.next().numpy()

    imgs_A = imgs_A/127.5- 1.0
    imgs_B = imgs_B/127.5 -1.0


    # Demo (for GIF)
    #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
    #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

    # Translate images to the other domain
    fake_B = generator_A2B.predict(imgs_A)
    fake_A = generator_B2A.predict(imgs_B)
    # Translate back to original domain
    reconstr_A = generator_B2A.predict(fake_B)
    reconstr_B = generator_A2B.predict(fake_A)

    gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Original', 'Translated', 'Reconstructed']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%s/%d_%d.png" % ('zebra2horse', epoch, batch_i))
    plt.close()
