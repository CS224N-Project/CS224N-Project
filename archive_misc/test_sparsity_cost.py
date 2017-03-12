import numpy as np
import tensorflow as tf

SIZE = 3
sparsity_factor = 0.0003
coherent_ratio = 2.0
coherent_factor = sparsity_factor * coherent_ratio

# tensorflow (dtypes and shapes required)
Z = tf.placeholder(dtype=tf.float32, shape=(SIZE,SIZE))

Zsum = tf.reduce_sum(Z, axis=0)
Zdiff = tf.reduce_sum(tf.abs(Z[1:]-Z[:-1]), axis=0)
sparsity_cost = tf.reduce_mean(Zsum) * sparsity_factor + tf.reduce_mean(Zdiff) * coherent_factor

z = np.eye(SIZE, dtype=np.float32)

print '\n tensorflow:'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print ''
    print(sess.run(sparsity_cost, feed_dict={ Z: z}))
    print ''
    print '\nz:'
    print(sess.run(Z, feed_dict={ Z: z}))
    print '\nzsum:'
    print(sess.run(Zsum, feed_dict={ Z: z}))
    print '\nzdiff:'
    print(sess.run(Zdiff, feed_dict={ Z: z}))
    print '\nsparsity_cost:'
    print(sess.run(sparsity_cost, feed_dict={ Z: z}))
    print ''

print '\n numpy:'

zsum = np.sum(z, axis=0)
zdiff = np.sum(np.abs(z[1:]-z[:-1]), axis=0)
sparsity_cost = np.mean(zsum) * sparsity_factor + np.mean(zdiff) * coherent_factor

print '\nz:'
print z
print '\nzsum:'
print zsum
print '\nzdiff:'
print zdiff
print '\nsparsity_cost:'
print sparsity_cost
print ''




