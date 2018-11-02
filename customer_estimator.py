import tensorflow as tf
import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import utils
import iris_data
from load_eeg import*
import load_eeg
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import to_categorical



def my_model(features, labels, mode, params):

    weight = features['weight']

    net = tf.feature_column.input_layer(features, params['feature_columns'])

    regularizer = tf.contrib.layers.l2_regularizer(scale=params['C'])

    for units in params['hidden_units']:
        print('making units {}'.format(units))
        net = tf.layers.dense(net, units = units, activation = tf.nn.relu, kernel_regularizer=regularizer)

    logits = tf.layers.dense(net, params['n_classes'], kernel_regularizer=regularizer, activation = None)

    predicted_classes = tf.argmax(logits,1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'class_ids':predicted_classes[:, tf.newaxis],
                       'probabilities': tf.nn.softmax(logits),
                       'logits':logits}
        return tf.estimator.EstimatorSpec(mode, predictions = predictions)

    print('pass here')
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits, weights = weight) # how to modify weights here

    print('pass here')

    accuracy = tf.metrics.accuracy(labels = labels, predictions= predicted_classes, name = 'acc_op')
    auc = tf.metrics.auc(labels = labels, predictions=tf.nn.softmax(logits)[:,1], name = 'auc')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('auc', auc[1])


    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss = loss, eval_metric_ops = metrics)


    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)




if __name__ == '__main__':



    subject = 'R1401J'
    sessions = get_sessions(subject, 'FR1')

    probs = []
    y_vec = []
    for session in sessions:

        (train_x,train_y),(test_x,test_y) = load_data(subject, session, 'FR1')
        train_y, test_y = train_y[0], test_y[0]

        my_feature_columns = []
        train_x.columns = [str(x) for x in train_x.keys()]
        test_x.columns = [str(x) for x in train_x.keys()]
        for key in train_x.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key))

        weight_column = tf.feature_column.numeric_column('weight')

        class_weights = compute_class_weight('balanced', np.unique(train_y), train_y)
        weights = np.zeros(len(train_y))

        pos_mask = train_y == 1
        weights[pos_mask] = class_weights[1]
        weights[~pos_mask] = class_weights[0]


        # train_y, test_y = to_categorical(train_y), to_categorical(test_y)


        my_checkpointing_config = tf.estimator.RunConfig(
            save_summary_steps=10,  # Save checkpoints every 20 minutes.
            keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
            log_step_count_steps= 1000
        )

        classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [],
            # The model must choose between 3 classes.
            'n_classes': 2,
            'C':7.2e-4, 'weight_column': weight_column}
        , model_dir='custom_' + str(session), config=my_checkpointing_config)

        # classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[], n_classes = 2, weight_column = weight_column,
        #                                         activation_fn=tf.nn.softmax,
        #                                         model_dir='savedir_' + str(session), config = my_checkpointing_config)

        train_x['weight'] = weights
        test_x['weight'] = np.ones(len(test_y))


        train_input_fn = tf.estimator.inputs.pandas_input_fn(x = train_x, y = train_y, shuffle=True, batch_size=12, num_epochs=1000)
        eval_input_fn = tf.estimator.inputs.pandas_input_fn(x = test_x, y = test_y, shuffle=False, num_epochs=1, batch_size=32)

        classifier.train(input_fn=train_input_fn,  max_steps=1000)

        # Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=eval_input_fn)

        predictions = classifier.predict(
            input_fn= eval_input_fn)


        proba = [x['probabilities'][1] for x in list(predictions)]
        probs.append(proba)
        y_vec.append(test_y)

    y_vec = np.concatenate(y_vec)
    probs = np.concatenate(probs)
    auc = roc_auc_score(y_vec, probs)
