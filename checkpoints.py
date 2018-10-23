import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import iris_data
from iris_data import*

(train_x, train_y), (test_x, test_y) = load_data()


def train_input_fn(features, labels, batch_size):

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)


my_feature_columns = []

for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key = key))


classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[10,10], n_classes = 3, model_dir= 'iris')

classifier.train(input_fn=lambda:train_input_fn(train_x, train_y, batch_size= 32), steps = 1000)

classifier.model_fn


eval_result = classifier.evaluate(input_fn= lambda:iris_data.eval_input_fn(test_x, test_y, 32))

expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

predictions = classifier.predict(input_fn=lambda:iris_data.eval_input_fn(predict_x,batch_size=32))

template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(template.format(iris_data.SPECIES[class_id],
                          100 * probability, expec))



