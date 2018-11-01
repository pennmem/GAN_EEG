# adversarial neural network for MNIST
import pandas as pd
import numpy as np
from load_eeg import load_data
import tensorflow as tf
tfgan = tf.contrib.gan
from tensorflow.contrib.training import HParams
from dc_gan import*
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist
tf.reset_default_graph()


(train_x,train_y),(test_x,test_y) = mnist.load_data()
n_rows = 28
n_cols = 28

train_x = train_x.reshape(-1,28*28)
train_x = train_x.astype(np.float32)
train_x = ((train_x - 128)/ 128)
#plot some images
# fig, axes = plt.subplots(nrows=2, ncols =2)
# for i in np.arange(2):
#     for j in np.arange(2):
#         axes[i,j].imshow(train_x[i*2+j,:,:])



test_x = test_x.astype(np.float16)

train_x_prior = np.random.uniform(-1.0,1.0,size = (60000,100)).astype(np.float32)

train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {'x':train_x, 'x_prior':train_x_prior}, y = train_y, shuffle=True, batch_size=32, num_epochs=10000)



my_checkpointing_config = tf.estimator.RunConfig(
    keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
    log_step_count_steps= 10)


hparams = {'n_output' : train_x.shape[1], 'generator_hidden_units' : [256,512], 'discriminator_hidden_units' : np.array([512,256]), 'generator_steps' :1, 'discriminator_steps': 2}
GAN_classifier = tf.estimator.Estimator(model_fn=model_fn,params=hparams, model_dir='GAN_MNIST_OUTPUT_NEW' , config=my_checkpointing_config)




for k in np.arange(0,11):

    # train
    GAN_classifier.train(input_fn=train_input_fn, steps=50)


    train_x_eval = np.random.uniform(-1.0,1.0,size = (100,100)).astype(np.float32)
    eval_input_fn =tf.estimator.inputs.numpy_input_fn(x = {'x':train_x[:100,], 'x_prior':train_x_eval}, shuffle=False,num_epochs=1)
    code = GAN_classifier.predict(input_fn=eval_input_fn)

    code = list(code)
    predictions = [x['code'] for x in code]
    predictions = np.array(predictions)
    predictions = predictions.reshape(-1,28,28)

    test = GAN_classifier.get_variable_value('Generator/dense/bias')



    # GAN_classifier.train(input_fn=train_input_fn, steps=2)
    # code1 = GAN_classifier.predict(input_fn=eval_input_fn)
    #
    # code1 = list(code1)
    # predictions1 = [x['code'] for x in code1]
    # predictions1 = np.array(predictions1)
    # predictions1 = predictions1.reshape(-1,28,28)
    #
    # test1 = GAN_classifier.get_variable_value('Generator/dense/bias')


        # plot some images
    fig, axes = plt.subplots(nrows=2, ncols =2)
    for i in np.arange(2):
        for j in np.arange(2):
            axes[i,j].imshow(predictions[i*2+j,:,:])

    fig.savefig("sample_cycle_" + str(100*k+1) + '.pdf')


    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # # generate sample image
    # fig, axes = plt.subplots(nrows=2,ncols=2)
    # for i in np.arange(4):
    #     j = i%2
    #     sns.distplot(train_x[i].values, color = 'red', ax = axes[int(i/2),j])
    #     sns.distplot(predictions[:,i], color = 'blue', ax = axes[int(i/2),j])
    #
