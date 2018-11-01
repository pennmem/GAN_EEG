# adversarial neural network for MNIST
import pandas as pd
import numpy as np
from load_eeg import load_data
import tensorflow as tf
tfgan = tf.contrib.gan
from tensorflow.contrib.training import HParams
from dc_gan import*
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
import os


mnist = tf.keras.datasets.mnist
tf.reset_default_graph()


(train_x,train_y),(test_x,test_y) = mnist.load_data()
n_rows = 28
n_cols = 28

train_x = train_x.astype(np.float32)
train_x = ((train_x - 128)/ 128)

train_x = train_x[:,:,:,np.newaxis]

#plot some images
# fig, axes = plt.subplots(nrows=2, ncols =2)
# for i in np.arange(2):
#     for j in np.arange(2):
#         axes[i,j].imshow(train_x[i*2+j,:,:])


input_dim = 100
test_x = test_x.astype(np.float16)
train_x_prior = np.random.uniform(-1.0,1.0,size = (60000,input_dim)).astype(np.float32)
n_features = train_x.shape[1]
n_sample = train_x.shape[0]

discriminator = discriminator_model()
discriminator_optimizer = Adam(0.00005,0.5)
discriminator.compile(loss = 'binary_crossentropy', optimizer=discriminator_optimizer, metrics = ['accuracy'])


discriminator.trainable = False
generator = generator_model()
generator_optimizer = Adam(0.0002,0.5)
input_noise = Input(shape = (input_dim,))
encoder_repr = generator(input_noise)
validity = discriminator(encoder_repr)
gan_model = Model(input_noise, outputs = [validity])
gan_model.compile(loss = 'binary_crossentropy', optimizer=generator_optimizer, metrics = ['accuracy'])


n_epochs = 1000
batch_size = 128


valid = np.ones((batch_size,1))
fake = np.zeros((batch_size,1))

d_loss_vec = []
g_loss_vec = []


for epoch in range(n_epochs):


    idx = np.random.randint(0, n_sample, batch_size)
    imgs = train_x[idx]

    for _ in np.arange(2):
        noise = np.random.normal(size = (batch_size, input_dim))
        latent_fake = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(latent_fake, fake)
        d_loss = 0.5*np.add(d_loss_fake , d_loss_real)


    noise = np.random.normal(size = (batch_size, input_dim))
    latent_fake = generator.predict(noise)
    g_loss = gan_model.train_on_batch(noise, valid)


    if epoch%20 == 0:
        print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, acc: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], 100*g_loss[1]))
        d_loss_vec.append(d_loss)
        g_loss_vec.append(g_loss)
        # generate fake images
        noise = np.random.normal(size = (10, input_dim))
        imgs_fake = generator.predict(noise)
        imgs_fake = imgs_fake.reshape(-1,28,28)

        fig, axes = plt.subplots(nrows=2, ncols =2)
        for i in np.arange(2):
            for j in np.arange(2):
                axes[i,j].imshow(imgs_fake[i+j,:,:])

        fig.savefig(os.getcwd() + "/results/sample_cycle_" + str(epoch) + '.pdf')


fig, ax = plt.subplots(1)
ax.plot(g_loss_vec)