# utility for gans
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Dropout
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
from tensorflow.contrib.gan.python.namedtuples import GANTrainSteps
from functools import partial


tfgan = tf.contrib.gan



# generator with hidden units and
def generator_fn(input_code, hidden_units, n_output):

    print(hidden_units)

    kwargs = {
    'kernel_regularizer': l2_regularizer(1000.0),
    'bias_regularizer': l2_regularizer(1000.0)}
    net = Dense(n_output, **kwargs)(input_code)
    if len(hidden_units) > 0:
        hidden_unit_left = hidden_units[1:]
        for hidden_unit in hidden_unit_left:
            net = Dense(hidden_unit, activation=tf.nn.leaky_relu, **kwargs)(net)
            net = Dropout(rate = 0.3)(net)

        net = Dense(n_output)(net)

    print(type(net))

    return net


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


def model_fn(features, labels, mode, params):


    x = features['x']
    rnd = features['x_prior']


    generator_fn_partial = partial(generator_fn, hidden_units = params['generator_hidden_units'], n_output = params['n_output'])

    discriminator_fn_partial = partial(discriminator_fn, hidden_units = params['discriminator_hidden_units'])

    gan_model = tfgan.gan_model(generator_fn = generator_fn_partial,
                                discriminator_fn = discriminator_fn_partial,
                                real_data = x,
                                generator_inputs = rnd)


    # cycle gan
    # improved_wgan_loss = tfgan.gan_loss(gan_model,
    # generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    # discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss)
    improved_wgan_loss = tfgan.gan_loss(gan_model,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss)


    with tf.variable_scope(gan_model.generator_scope, reuse=True):
        predictions = gan_model.generator_fn(rnd)


    if mode == tf.estimator.ModeKeys.TRAIN:
        generator_optimizer = tf.train.AdamOptimizer(0.01, beta1=0.5)
        discriminator_optimizer = tf.train.AdamOptimizer(0.01, beta1=0.5)
        gan_train_ops = tfgan.gan_train_ops(gan_model, improved_wgan_loss, generator_optimizer, discriminator_optimizer)
        gan_hooks = tfgan.get_sequential_train_hooks(GANTrainSteps(params['generator_steps'], params['discriminator_steps']))(gan_train_ops)
        return tf.estimator.EstimatorSpec(mode=mode, loss= improved_wgan_loss.discriminator_loss,
                                          train_op= gan_train_ops.global_step_inc_op,
                                          training_hooks=gan_hooks)


    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'code':predictions}

        return tf.estimator.EstimatorSpec(mode, predictions = predictions)


    else:
        eval_metrics_ops = {}
        return tf.estimator.EstimatorSpec(mode = mode, loss = improved_wgan_loss.discriminator_loss, eval_metrics_ops = eval_metrics_ops)
