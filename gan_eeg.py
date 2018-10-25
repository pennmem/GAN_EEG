# adversarial neural network for MNIST
import pandas as pd
import numpy as np
from load_eeg import load_data
import tensorflow as tf
tfgan = tf.contrib.gan
from tensorflow.contrib.training import HParams

from gan_utils import*


tf.reset_default_graph()


subject = 'R1001P'
session = 0
(train_x,train_y),(test_x,test_y) = load_data(subject, session, 'FR1')
train_y, test_y = train_y[0], test_y[0]


train_x_prior = np.random.uniform(-1.0,1.0,size = train_x.shape)

train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {'x':train_x.values, 'x_prior':train_x_prior}, y = train_y.values, shuffle=True, batch_size=32, num_epochs=1000)


train_x_eval = np.random.uniform(-1.0,1.0,size = train_x.shape)
eval_input_fn =tf.estimator.inputs.numpy_input_fn(x = {'x':train_x_eval, 'x_prior':train_x_eval}, shuffle=False,num_epochs=1)


my_checkpointing_config = tf.estimator.RunConfig(
    save_summary_steps=10,  # Save checkpoints every 20 minutes.
    keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
    log_step_count_steps= 100
)


hparams = {'n_output' : train_x.shape[1], 'generator_hidden_units' : [50], 'discriminator_hidden_units' : np.array([50]), 'generator_steps' :1, 'discriminator_steps': 1}


GAN_classifier = tf.estimator.Estimator(model_fn=model_fn,params=hparams, model_dir='CYCLE_GAN_OUTPUT_' + str(session), config=my_checkpointing_config)


for k in np.arange(0,11):

    # train
    GAN_classifier.train(input_fn=train_input_fn,  steps=1001)
    code = GAN_classifier.predict(input_fn=eval_input_fn)

    code = list(code)
    predictions = [x['code'] for x in code]
    predictions = np.array(predictions)

    import matplotlib.pyplot as plt
    import seaborn as sns

    # generate sample image
    fig, axes = plt.subplots(nrows=2,ncols=2)
    for i in np.arange(4):
        j = i%2
        sns.distplot(train_x[i].values, color = 'red', ax = axes[int(i/2),j])
        sns.distplot(predictions[:,i], color = 'blue', ax = axes[int(i/2),j])

    fig.savefig("sample_cycle_" + str(1000*k+1) + '.pdf')
