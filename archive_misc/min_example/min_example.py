#!/usr/bin/env python

#http://blog.aloni.org/posts/backprop-with-tensorflow/
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

a_0 = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

middle = 30
w_1 = tf.Variable(tf.truncated_normal([784, middle]))
b_1 = tf.Variable(tf.truncated_normal([1, middle]))
w_2 = tf.Variable(tf.truncated_normal([middle, 10]))
b_2 = tf.Variable(tf.truncated_normal([1, 10]))

# OUR CODE - BEGIN
# 784 = 28x28 pixel image (grayscale: each pixel value in [0.0,1.0])
# make zProbs trainable like w_1, etc. above:
zProbs = tf.Variable(tf.random_uniform(shape = [1, 784], minval=0.0, maxval=1.0))
zPreds = 1.0 / (1.0 + tf.exp(-60.0*(zProbs-0.5))) # sigmoid to simulate rounding
# zPreds = tf.select((zProbs >= 0.5), (zProbs + (1.0 - zProbs)), (zProbs - zProbs))
# zPreds = tf.floor(zProbs + 0.5)
#zPreds = zProbs
a_0_masked = tf.multiply(a_0, zPreds)
# OUR CODE - END

z_1 = tf.add(tf.matmul(a_0_masked, w_1), b_1)
a_1 = tf.sigmoid(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
a_2 = tf.sigmoid(z_2)

diff = tf.subtract(a_2, y)
cost = tf.multiply(diff, diff)

# OUR CODE - BEGIN
sparsity_factor = 0.0003
coherent_ratio = 2.0
coherent_factor = sparsity_factor * coherent_ratio
Zsum = tf.reduce_sum(zPreds, axis=1)
Zdiff = tf.reduce_sum(tf.abs(tf.subtract(zPreds[:,1:], zPreds[:,:-1])), axis=1)
selection_cost = tf.reduce_mean((Zsum * sparsity_factor) + (Zdiff * coherent_factor))
cost = cost + selection_cost
# OUR CODE - END

step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

zProbs_norm_list = []
w_1_norm_list = []
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for i in xrange(10000):
	    batch_xs, batch_ys = mnist.train.next_batch(10)
	    _, zProbs_, w_1_, zPreds_ = sess.run([step, zProbs, w_1, zPreds], feed_dict = {a_0: batch_xs,
	                                y : batch_ys})
	    zProbs_norm_list.append(np.linalg.norm(zProbs_))
	    w_1_norm_list.append(np.linalg.norm(w_1_))

	    if i % 1000 == 0:
	        res, cost_ = sess.run([acct_res, cost], feed_dict =
	                       {a_0: mnist.test.images[:1000],
	                        y : mnist.test.labels[:1000]})
	        print res
	        print np.linalg.norm(cost_)
	        print len(set(zProbs_norm_list))
	        print len(set(w_1_norm_list))
	        print ''
	        zProbs_norm_list = []
	        w_1_norm_list = []

plt.imsave('zPreds.png', np.array(zPreds_).reshape(28,28), cmap=cm.gray)

