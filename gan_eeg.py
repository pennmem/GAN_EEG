# adversarial neural network for MNIST
import pandas as pd
import numpy as np
from load_eeg import load_data
import tensorflow as tf
tfgan = tf.contrib.gan
from tensorflow.contrib.training import HParams

from gan_utils import*



subject = 'R1001P'
session = 0
(train_x,train_y),(test_x,test_y) = load_data(subject, session, 'FR1')
train_y, test_y = train_y[0], test_y[0]


train_x_prior = np.random.normal(0,1,size = train_x.shape)

train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {'x':train_x.values, 'x_prior':train_x_prior}, y = train_y.values, shuffle=True, batch_size=12, num_epochs=1000)


my_checkpointing_config = tf.estimator.RunConfig(
    save_summary_steps=10,  # Save checkpoints every 20 minutes.
    keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
    log_step_count_steps= 1000
)


hparams = {'n_output' : train_x.shape[1], 'generator_hidden_units' : [50,50], 'discriminator_hidden_units' : np.array([50,50]), 'generator_steps' :1, 'discriminator_steps': 1}


GAN_classifier = tf.estimator.Estimator(model_fn=model_fn,params=hparams, model_dir='GAN_OUTPUT_' + str(session), config=my_checkpointing_config)
GAN_classifier.train(input_fn=train_input_fn,  max_steps=100)
