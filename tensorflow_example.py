# tensorflow tutorial

import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import re
import matplotlib.pyplot as plt

# build computational graph
a = tf.constant(3.0, dtype = tf.float32)
b = tf.constant(4.0)
c = a + b
print(a)
print(b)
print(c)

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

sess = tf.Session()
print(sess.run(c))
print(sess.run({'ab':(a,b), 'c':c}))



vec = tf.random_uniform(shape = (3,3))
out1 = vec + 1
out2 = vec + 1
print(sess.run(out1))
print(sess.run(out2))

print(sess.run((out1,out2)))

# feeding
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

print(sess.run(z, feed_dict={x:10, y:20}))
print(sess.run(z, feed_dict={x:[1,2], y:[10,20]}))

# importing data
mydata = [[0,1], [2,3], [4,5], [6,7]]
slices = tf.data.Dataset.from_tensors(mydata)
next_item = slices.make_one_shot_iterator().get_next()
while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break



r = tf.random_normal(shape = [10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()
sess.run(iterator.initializer)

while True:
  try:
    print(sess.run(next_row))
  except tf.errors.OutOfRangeError:
    break

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4,10]))
print(dataset1.output_shapes)
print(dataset1.output_types)
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4,10], maxval = 100, dtype = tf.int32)))
print(dataset2.output_shapes)
dataset2.output_types

dataset3 = tf.data.Dataset.zip((dataset1,dataset2))

# creating iterator
data = np.arange(100)
dataset = tf.data.Dataset.from_tensor_slices(data)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
    value = sess.run(next_element)
    print(value)
    assert i == value


# layers

x = tf.placeholder(tf.float32, shape = [None, 3])
linear_model = tf.layers.Dense(units = 1)
y = linear_model(x)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, {x:[[1,2,3]]}))

features = {'sales': [[5],[10],[8],[9]], 'department':['sports', 'sports', 'gardening', 'gardening']}
department_column = tf.feature_column.categorical_column_with_vocabulary_list('department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)
columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))
print(sess.run(inputs))



# training model
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
linear_model = tf.layers.Dense(units = 1)
y_pred = linear_model(x)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y_pred))

loss = tf.losses.mean_squared_error(labels = y_true, predictions = y_pred)
print(sess.run(loss))


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(100):
    _, loss_value = sess.run((train,loss))
    print(loss_value)


mammal = tf.Variable("elephant", tf.string)
my_image = tf.zeros([10,299,299,3])

p = tf.placeholder(tf.float32)
t = p + 10.0
sess = tf.Session()
t.eval(session = sess, feed_dict = {p:32})

import iris_data
train, test = iris_data.load_data()
features, labels = train
batch_size = 100
iris_data.train_input_fn(features, labels, batch_size)
