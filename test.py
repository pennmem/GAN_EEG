import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-vfg","--verbosity", help = 'increase output verbosity', action = 'store_true')
parser.add_argument('square', help = 'display a square of a given number', type = int)
args = parser.parse_args()

answer = args.square**2
if args.verbosity:
    print("the square of {} equals {}".format(args.square, answer))
else:
    print(answer)



import tensorflow as tf
import numpy as np
import scipy
x = tf.placeholder(tf.float32, shape = (1024,1024))
y = tf.matmul(x,x)
with tf.Session() as sess:
    rand_array = np.random.rand(1024,1024)
    print(sess.run(y, feed_dict= {x:rand_array}).apply(scipy.stats.describe))