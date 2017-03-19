
import tensorflow as tf

# constant = tf.Variable("This is constant")


# saver = tf.train.Saver()

# init = tf.global_variables_initializer()



# with tf.Session() as sess:

#     sess.run(init)
#     saver.restore(sess, './sample_code')
#     for i in range(4):
#         print(sess.run(constant))

#     saver.save(sess, 'sample_code')


constant = tf.Variable("This is constant")
saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, './generator.weights')
    all_vars = tf.get_collection('vars')
    for v in all_vars:
    	v_ = sess.run(v)
    	print(v_)


# sess = tf.Session()

# new_saver = tf.train.import_meta_graph('./generator.weights', clear_devices=True)
# # new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
# new_saver.restore(sess, './generator.weights')
# all_vars = tf.get_collection('vars')
# for v in all_vars:
#     v_ = sess.run(v)
#     print(v_)


# for i, (test_x, test_y, test_sentLen, test_mask, test_rat) in enumerate(
#     get_minibatches_test(self.test_x, self.test_y, self.test_sentLen,
#                          self.test_mask, self.rationals,
#                          self.config.batch_size, False)):
#     feed = self.create_feed_dict(inputs_batch=test_x,
#                                  mask_batch=test_mask,
#                                  seqLen=test_sentLen,
#                                  labels_batch=test_y,
#                                  dropout=self.config.drop_out,
#                                  l2_reg=self.config.l2Reg,
#                                  rationals=test_rat)
#     preds = sess.run(self.zPreds, feed_dict = feed)
#     np.savetxt(outFile, preds, delimiter=' ')

