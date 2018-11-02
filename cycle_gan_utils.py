# utility for gans
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras import Sequential, Input, Model
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
from tensorflow.contrib.gan.python.namedtuples import GANTrainSteps
from functools import partial

# generator with hidden units and
def generator_fn(input_code, hidden_units, n_output):

    kwargs = {
    'kernel_regularizer': l2_regularizer(1.0e-5),
    'bias_regularizer': l2_regularizer(1.0e-5)}
    net = Dense(n_output, activation=tf.nn.leaky_relu)(input_code)
    # net = Dense(n_output, **kwargs)(input_code)
    if len(hidden_units) > 0:
        hidden_unit_left = hidden_units[1:]
        for hidden_unit in hidden_unit_left:
            net = Dense(hidden_unit, activation=tf.nn.leaky_relu, **kwargs)(net)
            net = BatchNormalization(momentum = 0.8)(net)
            net = Dropout(rate = 0.3)(net)

        net = Dense(n_output, activation=tf.nn.tanh)(net)

    print(type(net))

    return net


def discriminator_fn(code, hidden_units):

    print('input code dimension {}'.format(code.shape))
    # print('input code2 dimension {}'.format(code2.shape))


    kwargs = {
    'kernel_regularizer': l2_regularizer(1.0e-5),
    'bias_regularizer': l2_regularizer(1.0e-5)}


    h = code

    for hidden_unit in hidden_units:
        # h = Dense(hidden_unit, activation=tf.nn.leaky_relu, **kwargs)(h)
        h = Dense(hidden_unit, activation=tf.nn.leaky_relu)(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Dropout(rate = 0.3)(h)

    h = Dense(1, name = 'dis_out')(h)
    # h = Dense(1, name = 'dis_out', **kwargs)(h)
    logits = h


    print(type(logits))
    return logits


def build_discriminator(latent_dim, hidden_units):

    kwargs = {
    'kernel_regularizer': l2_regularizer(1.0e-5),
    'bias_regularizer': l2_regularizer(1.0e-5)}


    model = Sequential()
    model.add(Dense(hidden_units[0], input_dim=latent_dim))
    model.add(LeakyReLU(alpha =0.2))
    model.add(Dropout(0.4))

    for hidden_unit in hidden_units:
        model.add(Dense(hidden_unit, **kwargs))
        model.add(LeakyReLU(alpha =0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))

    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    encoded_repr = Input(shape = (latent_dim,))
    validity = model(encoded_repr)

    model = Model(encoded_repr, validity)
    model.summary()

    return model

def build_generator(input_dim, hidden_units, n_output):


    kwargs = {
    'kernel_regularizer': l2_regularizer(1.0e-5),
    'bias_regularizer': l2_regularizer(1.0e-5)}


    print(hidden_units)
    model = Sequential()
    model.add(Dense(hidden_units[0], input_dim=input_dim))
    model.add(LeakyReLU(alpha =0.2))
    model.add(Dropout(0.4))


    for hidden_unit in hidden_units[1:]:
        print(hidden_unit)
        model.add(Dense(hidden_unit))
        model.add(LeakyReLU(alpha =0.2))
        model.add(Dropout(0.4))
        model.add(BatchNormalization(momentum=0.8))


    model.add(Dense(n_output, activation = 'tanh'))
    encoded_repr = Input(shape = (input_dim,))
    validity = model(encoded_repr)

    model = Model(encoded_repr, validity)
    model.summary()

    return model

