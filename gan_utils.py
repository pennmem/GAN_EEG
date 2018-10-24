# utility for gans
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
from tensorflow.contrib.gan.python.namedtuples import GANTrainSteps
from functools import partial


tfgan = tf.contrib.gan



# generator with hidden units and
def generator_fn(input_code, hidden_units, n_output):

    print(hidden_units)

    kwargs = {
    'kernel_regularizer': l2_regularizer(1e-6),
    'bias_regularizer': l2_regularizer(1e-6)}
    if len(hidden_units) == 0:
        net = Dense(n_output, **kwargs)(input_code)
    else:
        net = Dense(hidden_units[0])(input_code)
        hidden_unit_left = hidden_units[1:]
        for hidden_unit in hidden_unit_left:
            net = Dense(hidden_unit, activation=tf.nn.leaky_relu, **kwargs)(net)

        net = Dense(n_output, **kwargs)(net)

    print(type(net))

    return net


def discriminator_fn(code, code2, hidden_units):

    print(hidden_units)

    kwargs = {
    'kernel_regularizer': l2_regularizer(1e-6),
    'bias_regularizer': l2_regularizer(1e-6)}
    h = code

    for hidden_unit in hidden_units:
        h = Dense(hidden_unit, activation=tf.nn.leaky_relu, **kwargs)(h)

    h = Dense(1, name = 'dis_out', **kwargs)(h)
    logits = h

    return logits


def model_fn(features, labels, mode, params):

    if mode == tf.estimator.ModeKeys.PREDICT:
        raise NotImplementedError()  # raise error for predict

    else:
        x = features['x']
        rnd = features['x_prior']

        print(x.shape)
        print(rnd.shape)

        generator_fn_partial = partial(generator_fn, hidden_units = params['generator_hidden_units'], n_output = params['n_output'])

        discriminator_fn_partial = partial(discriminator_fn, hidden_units = params['discriminator_hidden_units'])

        gan_model = tfgan.gan_model(generator_fn = generator_fn_partial,
                                    discriminator_fn = discriminator_fn_partial,
                                    real_data = x,
                                    generator_inputs = rnd)


        improved_wgan_loss = tfgan.gan_loss(gan_model,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss)


        if mode == tf.estimator.ModeKeys.TRAIN:
            generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
            discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
            gan_train_ops = tfgan.gan_train_ops(gan_model, improved_wgan_loss, generator_optimizer, discriminator_optimizer)
            gan_hooks = tfgan.get_sequential_train_hooks(GANTrainSteps(params['generator_steps'], params['discriminator_steps']))(gan_train_ops)
            return tf.estimator.EstimatorSpec(mode=mode, loss= improved_wgan_loss.discriminator_loss,
                                              train_op= gan_train_ops.global_step_inc_op,
                                              training_hooks=gan_hooks)

        else:
            eval_metrics_ops = {}
            return tf.estimator.EstimatorSpec(mode = mode, loss = improved_wgan_loss.discriminator_loss, eval_metrics_ops = eval_metrics_ops)
