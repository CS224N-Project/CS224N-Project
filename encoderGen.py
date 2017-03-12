import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from preprocess import readOurData
from model import Model
import time

# from util import print_sentence, write_conll, read_conll
from data_util import load_and_preprocess_data
from utils.general_utils import Progbar
from utils.parser_utils import minibatches, load_and_preprocess_data
from config import Config


#################
### RNN Model ###
#################

class RNNEncoderModel(object):
    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        self.eval = self.evaluate(self.pred)

    def _read_data(self, train_path, dev_path, embedding_path):
        '''
        Helper function to read in our data. Used to construct our RNNModel
        :param train_path: path to training data
        :param dev_path: path to development data
        :param embedding_path: path to embeddings
        :return: read in training/development data with padding masks and
        embedding dictionaries
        '''
        from preprocess import readOurData
        train_x_pad, train_y, train_mask, train_sentLen, dev_x_pad, dev_y, dev_mask, dev_sentLen, embeddingDictPad = readOurData(
            train_path, dev_path, embedding_path)
        return train_x_pad, train_y, train_mask, train_sentLen, dev_x_pad, dev_y, dev_mask, dev_sentLen, embeddingDictPad

    def add_placeholders(self):
        # batchSize X sentence X numClasses
        self.inputPH = tf.placeholder(dtype=tf.int32,
                                      shape=(None, self.config.max_sentence),
                                      name='input2')
        # batchSize X numClasses
        self.labelsPH = tf.placeholder(dtype=tf.float32,
                                       shape=(None, self.config.n_class),
                                       name='labels2')
        # mask over sentences not long enough
        self.maskPH = tf.placeholder(dtype=tf.bool,
                                     shape=(None, self.config.max_sentence),
                                     name='mask2')
        self.dropoutPH = tf.placeholder(dtype=tf.float32,
                                        shape=(),
                                        name='dropout2')
        self.seqPH = tf.placeholder(dtype=tf.float32,
                                    shape=(None,),
                                    name='sequenceLen2')
        self.l2RegPH = tf.placeholder(dtype=tf.float32,
                                      shape=(),
                                      name='l2Reg2')

    def create_feed_dict(self, inputs_batch, mask_batch, sentLen,
                         labels_batch=None, dropout=1.0, l2_reg=0.0):

        feed_dict = {
            self.inputPH: inputs_batch,
            self.maskPH: mask_batch,
            self.dropoutPH: dropout,
            self.l2RegPH: l2_reg,
            self.seqPH: sentLen
        }

        # Add labels if not none
        if labels_batch is not None:
            feed_dict[self.labelsPH] = labels_batch

        return feed_dict

    def add_embedding(self):
        embedding_shape = (-1,
                           self.config.max_sentence,
                           self.config.embedding_size)

        pretrainEmbeds = tf.Variable(self.pretrained_embeddings,
                                     dtype=tf.float32)
        embeddings = tf.nn.embedding_lookup(pretrainEmbeds, self.inputPH)
        embeddings = tf.reshape(embeddings, shape=embedding_shape)

        return embeddings

    def add_prediction_op(self):

        # get relevent embedding data
        x = self.add_embedding()
        currBatch = tf.shape(x)[0]

        # Extract sizes
        hidden_size = self.config.hidden_size
        n_class = self.config.n_class
        batch_size = self.config.batch_size
        max_sentence = self.config.max_sentence
        embedding_size = self.config.embedding_size


        # Define our prediciton layer variables
        M = tf.get_variable(name = 'W_blah',
                            shape = ((2 * hidden_size), n_class),
                            dtype = tf.float32,
                            initializer = tf.contrib.layers.xavier_initializer())

        a = tf.get_variable(name = 'b_blah',
                            shape = (n_class,),
                            dtype = tf.float32,
                            initializer = tf.constant_initializer(0.0))




        cell1 = tf.nn.rnn_cell.BasicRNNCell(hidden_size, activation = tf.tanh)
        cell2 = tf.nn.rnn_cell.BasicRNNCell(hidden_size, activation = tf.tanh)

        cell1_drop = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=self.dropoutPH)
        cell2_drop = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=self.dropoutPH)
        cell_multi = tf.nn.rnn_cell.MultiRNNCell([cell1_drop, cell2_drop])
        with tf.variable_scope('secondENcoder'):
            result = tf.nn.dynamic_rnn(cell_multi, x, dtype = tf.float32, sequence_length = self.seqPH)
        h_t = tf.concat(concat_dim = 1, values = [result[1][0], result[1][1]])

        y_t = tf.tanh(tf.matmul(h_t, M) + a)

        return y_t

    def add_prediction_op2(self, data):

        # get relevent embedding data
        currBatch = tf.shape(data)[0]

        # Extract sizes
        hidden_size = self.config.hidden_size
        n_class = self.config.n_class
        batch_size = self.config.batch_size
        max_sentence = self.config.max_sentence
        embedding_size = self.config.embedding_size

        # Define our prediciton layer variables
        W = tf.get_variable(name='W',
                            shape=((2 * hidden_size), n_class),
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(name='b',
                            shape=(n_class,),
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))

        cell1 = tf.nn.rnn_cell.BasicRNNCell(embedding_size, activation=tf.tanh)
        cell2 = tf.nn.rnn_cell.BasicRNNCell(hidden_size, activation=tf.tanh)

        cell1_drop = tf.nn.rnn_cell.DropoutWrapper(cell1,
                                                   output_keep_prob=self.dropoutPH)
        cell2_drop = tf.nn.rnn_cell.DropoutWrapper(cell2,
                                                   output_keep_prob=self.dropoutPH)
        cell_multi = tf.nn.rnn_cell.MultiRNNCell([cell1_drop, cell2_drop])
        result = tf.nn.dynamic_rnn(cell_multi,
                                   data,
                                   dtype=tf.float32,
                                   sequence_length=self.seqPH)
        h_t = tf.concat(concat_dim=1,
                        values=[result[1][0], result[1][1]])

        y_t = tf.tanh(tf.matmul(h_t, W) + b)

        return y_t

    def add_loss_op(self, pred):
        # Compute L2 loss
        L2loss = tf.nn.l2_loss(self.labelsPH - pred)
        L2loss = tf.reduce_mean(L2loss)

        # Apply L2 regularization
        reg_by_var = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
        regularization = tf.reduce_sum(reg_by_var)

        loss = (10.0 * L2loss) + (self.l2RegPH * regularization)
        return loss

    def add_training_op(self, loss):
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        train_op = opt.minimize(loss)
        return train_op

    ## TODO: Add def evaluate(test_set)
    def evaluate(self, pred):
        diff = self.labelsPH - pred
        prod = tf.matmul(diff, diff, transpose_a=True)
        se = tf.reduce_sum(prod)
        return se

    def evaluate_on_batch(self, sess, inputs_batch, labels_batch, mask_batch,
                          sentLen):
        feed = self.create_feed_dict(inputs_batch=inputs_batch,
                                     mask_batch=mask_batch,
                                     sentLen=sentLen,
                                     labels_batch=labels_batch,
                                     dropout=self.config.drop_out,
                                     l2_reg=self.config.l2Reg)
        se = sess.run(self.eval, feed_dict=feed)
        return se

    ### NO NEED TO UPDATE BELOW
    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch,
                       sentLen):
        feed = self.create_feed_dict(inputs_batch=inputs_batch,
                                     mask_batch=mask_batch,
                                     sentLen=sentLen,
                                     labels_batch=labels_batch,
                                     dropout=self.config.drop_out,
                                     l2_reg=self.config.l2Reg)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess):
        train_se = 0.0
        prog = Progbar(
            target=1 + self.train_x.shape[0] / self.config.batch_size)
        for i, (train_x, train_y, train_sentLen, mask) in enumerate(
                minibatches(self.train_x, self.train_y, self.train_sentLen,
                            self.train_mask, self.config.batch_size)):
            loss = self.train_on_batch(sess, train_x, train_y, mask,
                                       train_sentLen)
            train_se += self.evaluate_on_batch(sess, train_x, train_y, mask,
                                               train_sentLen)
            prog.update(i + 1, [("train loss", loss)])

        train_obs = self.train_x.shape[0]
        train_mse = train_se / train_obs

        print 'Training MSE is {0}'.format(train_mse)

        print "Evaluating on dev set",
        dev_se = 0.0
        for i, (dev_x, dev_y, dev_sentLen, dev_mask) in enumerate(
                minibatches(self.dev_x, self.dev_y, self.dev_sentLen,
                            self.dev_mask, self.config.batch_size)):
            dev_se += self.evaluate_on_batch(sess, dev_x, dev_y, dev_mask,
                                             dev_sentLen)

        dev_obs = self.dev_x.shape[0]
        dev_mse = dev_se / dev_obs

        print "- dev MSE: {:.2f}".format(dev_mse)
        return dev_mse

    # def run_epoch(self, sess, parser, train_examples, dev_set):
    #     prog = Progbar(target=1 + len(train_examples) / self.config.batch_size)
    #     for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
    #         loss = self.train_on_batch(sess, train_x, train_y)
    #         prog.update(i + 1, [("train loss", loss)])
    #
    #     print "Evaluating on dev set",
    #     dev_UAS, _ = parser.parse(dev_set)
    #     print "- dev UAS: {:.2f}".format(dev_UAS * 100.0)
    #     return dev_UAS

    def fit(self, sess, saver):
        best_dev_mse = np.inf
        for epoch in range(self.config.epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.epochs)
            dev_mse = self.run_epoch(sess)
            if dev_mse < best_dev_mse:
                best_dev_mse = dev_mse
                if saver:
                    print "New best dev MSE! Saving model in ./encoder.weights"
                    # saver.save(sess, './encoder.weights', write_meta_graph = False)
                    saver.save(sess, './encoder.weights')
            print

    ## add def eval here

    def __init__(self, config, embedding_path, train_path, dev_path):
        train_x_pad, train_y, train_mask, train_sentLen, dev_x_pad, dev_y, dev_mask, dev_sentLen, embeddingDictPad = self._read_data(
            train_path, dev_path, embedding_path)
        self.train_x = train_x_pad
        self.train_y = train_y
        self.train_mask = train_mask
        self.train_sentLen = train_sentLen
        self.dev_x = dev_x_pad
        self.dev_y = dev_y
        self.dev_mask = dev_mask
        self.dev_sentLen = dev_sentLen
        self.pretrained_embeddings = embeddingDictPad
        self.maskId = len(embeddingDictPad) - 1
        # Update our config with data parameters
        self.config = config
        self.config.max_sentence = max(train_x_pad.shape[1], dev_x_pad.shape[1])
        # self.config.max_sentence = train_x_pad.shape[1]
        self.config.n_class = train_y.shape[1]
        self.config.embedding_size = embeddingDictPad.shape[1]
        self.build()