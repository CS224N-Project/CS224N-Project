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
from rnncell import RNNCell
from config import Config
from encoderGen import RNNEncoderModel

'''
Set up classes and functions
'''

class RNNGeneratorModel(object):
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
                                      shape=(None,
                                             self.config.max_sentence),
                                      name='input')
        # batchSize X numClasses
        self.labelsPH = tf.placeholder(dtype=tf.float32,
                                       shape=(None,
                                              self.config.n_class),
                                       name='labels')
        # mask over sentences not long enough
        self.maskPH = tf.placeholder(dtype=tf.bool,
                                     shape=(None,
                                            self.config.max_sentence),
                                     name='mask')
        self.dropoutPH = tf.placeholder(dtype=tf.float32,
                                        shape=(),
                                        name='dropout')
        self.seqPH = tf.placeholder(dtype=tf.int32,
                                    shape=(None,),
                                    name='sequenceLen')
        self.l2RegPH = tf.placeholder(dtype=tf.float32,
                                      shape=(),
                                      name='l2Reg')

    def create_feed_dict(self, inputs_batch, mask_batch, seqLen, labels_batch=None,
                         dropout=1.0, l2_reg=0.0):

        feed_dict = {
            self.inputPH: inputs_batch,
            self.maskPH: mask_batch,
            self.seqPH: seqLen,
            self.dropoutPH: dropout,
            self.l2RegPH: l2_reg
        }

        # Add labels if not none
        if labels_batch is not None:
            feed_dict[self.labelsPH] = labels_batch

        return feed_dict

    def add_embedding(self):
        embedding_shape = (-1,
                           self.config.max_sentence,
                           self.config.embedding_size)

        pretrainEmbeds = tf.constant(self.pretrained_embeddings,
                                     dtype=tf.float32)
        embeddings = tf.nn.embedding_lookup(pretrainEmbeds, self.inputPH)
        embeddings = tf.reshape(embeddings, shape=embedding_shape)

        return embeddings

    def add_prediction_op(self):

        # get relevent embedding data
        x = self.add_embedding()
        currBatch = tf.shape(x)[0]
        xDrop = tf.nn.dropout(x, self.dropoutPH)
        xRev = tf.reverse(xDrop, dims = [False, True, False])
        # embeds = tf.concat(concat_dim=1, values = [xDrop, xRev])

        # Extract sizes
        hidden_size = self.config.hidden_size
        n_class = self.config.n_class
        batch_size = self.config.batch_size
        max_sentence = self.config.max_sentence
        embedding_size = self.config.embedding_size

        # Define internal RNN Cells
        genCell1Layer1 = tf.nn.rnn_cell.BasicRNNCell(num_units = hidden_size,
                                                     # input_size = embedding_size,
                                                     activation = tf.tanh)
        genCell2Layer1 = tf.nn.rnn_cell.BasicRNNCell(num_units = hidden_size,
                                                     # input_size = embedding_size,
                                                     activation = tf.tanh)
        # genCell1Layer2 = tf.nn.rnn_cell.BasicRNNCell(num_units = hidden_size,
        #                                              # input_size = hidden_size,
        #                                              activation = tf.tanh)
        # genCell2Layer2 = tf.nn.rnn_cell.BasicRNNCell(num_units = hidden_size,
        #                                              # input_size = hidden_size,
        #                                              activation = tf.tanh)

        # Apply dropout to each cell
        genC1L1Drop = tf.nn.rnn_cell.DropoutWrapper(genCell1Layer1,
                                                    output_keep_prob=self.dropoutPH)
        genC2L1Drop = tf.nn.rnn_cell.DropoutWrapper(genCell2Layer1,
                                                    output_keep_prob=self.dropoutPH)
        # genC1L2Drop = tf.nn.rnn_cell.DropoutWrapper(genCell1Layer2,
        #                                             output_keep_prob=self.dropoutPH)
        # genC2L2Drop = tf.nn.rnn_cell.DropoutWrapper(genCell2Layer2,
        #                                             output_keep_prob=self.dropoutPH)

        # Stack each for multi Cell
        # multiFwd = tf.nn.rnn_cell.MultiRNNCell([genC1L1Drop, genC1L2Drop])
        # multiBwd = tf.nn.rnn_cell.MultiRNNCell([genC2L1Drop, genC2L2Drop])

        # Set inital states
        # fwdInitState = genC1L1Drop.zero_state(batch_size = currBatch,
        #                                    dtype = tf.float32)
        # bwdInitState = genC2L1Drop.zero_state(batch_size = currBatch,
        #                                    dtype = tf.float32)

        fwdInitState = genCell1Layer1.zero_state(batch_size = currBatch,
                                                 dtype = tf.float32)
        bwdInitState = genCell2Layer1.zero_state(batch_size = currBatch,
                                                 dtype = tf.float32)

        _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw = genC1L1Drop,
                                                    cell_bw = genC2L1Drop,
                                                    inputs = x,
                                                    initial_state_fw = fwdInitState,
                                                    initial_state_bw = bwdInitState,
                                                    dtype = tf.float32,
                                                    sequence_length = self.seqPH
                                                    )

        # states returns tuple (fwdState, bwdState) where each is a 3-d tensor
        # of (depth, batchsize, hiddendim). unpack axis 0 to get each final state
        # unpackedStates1 = tf.unpack(states[0], axis = 0)
        # unpackedStates2 = tf.unpack(states[1], axis = 0)
        # states = unpackedStates1 + unpackedStates2

        finalStates = tf.concat(concat_dim = 1, values = states)
        # finalStates = states
        # finalStatesIn = tf.shape(finalStates)[1]

        # Define our prediciton layer variables
        U = tf.get_variable(name='W_gen',
                            shape=((2 * hidden_size), self.config.max_sentence),
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        c = tf.get_variable(name='b_gen',
                            shape=(self.config.max_sentence,),
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))

        # zLayer probabilities - each prob is prob of keeping word in review
        zProbs = tf.sigmoid(tf.matmul(finalStates, U) + c)
        # zProbs = tf.stop_gradient(zProbs)

        # sample zprobs to pick which review words to keep. mask unselected words
        uniform = tf.random_uniform(shape = tf.shape(zProbs), minval=0, maxval=1) < zProbs
        # uniform = tf.stop_gradient(
        #     tf.random_uniform(shape=tf.shape(zProbs), minval=0,
        #                       maxval=1) < zProbs, 'uniform')
        self.zPreds = tf.select(uniform,
                                tf.ones(shape = tf.shape(uniform), dtype = tf.float32),
                                tf.zeros(shape = tf.shape(uniform), dtype = tf.float32))
        masks = tf.zeros(shape = tf.shape(zProbs), dtype = tf.int32) + self.maskId
        maskedInputs = tf.select(uniform, self.inputPH, masks)

        # Return masked embeddings to pass to encoder
        embedding_shape = (-1,
                           self.config.max_sentence,
                           self.config.embedding_size)

        maskedEmbeddings = tf.nn.embedding_lookup(self.pretrained_embeddings,
                                                  maskedInputs)
        maskedEmbeddings = tf.cast(maskedEmbeddings, tf.float32)
        maskedEmbeddings = tf.reshape(maskedEmbeddings, shape=embedding_shape)

        # Use encoder to make predictions
        # encoderPreds = self.encoder.add_prediction_op2(maskedEmbeddings)

        # Define our prediciton layer variables
        W = tf.get_variable(name='W',
                            shape=(hidden_size, n_class),
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(name='b',
                            shape=(n_class,),
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))

        cell1 = tf.nn.rnn_cell.BasicRNNCell(embedding_size, activation=tf.tanh)
        # cell2 = tf.nn.rnn_cell.BasicRNNCell(hidden_size, activation=tf.tanh)

        cell1_drop = tf.nn.rnn_cell.DropoutWrapper(cell1,
                                                   output_keep_prob=self.dropoutPH)
        # cell2_drop = tf.nn.rnn_cell.DropoutWrapper(cell2,
        #                                            output_keep_prob=self.dropoutPH)
        # cell_multi = tf.nn.rnn_cell.MultiRNNCell([cell1_drop, cell2_drop])
        result = tf.nn.dynamic_rnn(cell1,
                                   maskedEmbeddings,
                                   dtype=tf.float32,
                                   sequence_length=self.seqPH)
        # h_t = tf.concat(concat_dim=1,
        #                 values=[result[1][0], result[1][1]])

        y_t = tf.tanh(tf.matmul(result[1], W) + b)

        return y_t

    def add_loss_op(self, pred):
        # Compute L2 loss
        L2loss = tf.nn.l2_loss(self.labelsPH - pred)
        L2loss = tf.reduce_mean(L2loss)

        # Apply L2 regularization - all vars
        reg_by_var = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
        regularization = tf.reduce_sum(reg_by_var)

        # apply L2 regularization to number of predictions
        # regPreds = tf.reduce_sum(tf.nn.l2_loss(self.zPreds))

        # apply reg to sequence
        sparsity_factor = 0.0003
        coherent_ratio = 2.0
        coherent_factor = sparsity_factor * coherent_ratio
        Zsum = tf.reduce_sum(self.zPreds, axis=0)
        Zdiff = tf.reduce_sum(tf.abs(self.zPreds[1:] - self.zPreds[:-1]), axis=0)
        sparsity_cost = tf.reduce_mean(Zsum) * sparsity_factor + tf.reduce_mean(
            Zdiff) * coherent_factor

        loss = (10.0 * L2loss) + (self.l2RegPH * regularization) + sparsity_cost

        return loss

    def add_training_op(self, loss):
        opt = tf.train.AdamOptimizer(learning_rate = self.config.lr)
        train_op = opt.minimize(loss)
        return train_op

    ## TODO: Add def evaluate(test_set)
    def evaluate(self, pred):
        diff = self.labelsPH - pred
        prod = tf.matmul(diff, diff, transpose_a=True)
        se = tf.reduce_sum(prod)
        return se

    def evaluate_on_batch(self, sess, inputs_batch, labels_batch, mask_batch, sentLen):
        feed = self.create_feed_dict(inputs_batch = inputs_batch,
                                     mask_batch = mask_batch,
                                     seqLen = sentLen,
                                     labels_batch=labels_batch,
                                     dropout=self.config.drop_out,
                                     l2_reg=self.config.l2Reg)
        se = sess.run(self.eval, feed_dict=feed)
        return se

    ### NO NEED TO UPDATE BELOW
    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch, sentLen):
        feed = self.create_feed_dict(inputs_batch = inputs_batch,
                                     mask_batch = mask_batch,
                                     seqLen=sentLen,
                                     labels_batch=labels_batch,
                                     dropout=self.config.drop_out,
                                     l2_reg=self.config.l2Reg)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess):
        train_se = 0.0
        prog = Progbar(target=1 + self.train_x.shape[0] / self.config.batch_size)
        for i, (train_x, train_y, train_sentLen, mask) in enumerate(minibatches(self.train_x, self.train_y, self.train_sentLen, self.train_mask, self.config.batch_size)):
            loss = self.train_on_batch(sess, train_x, train_y, mask, train_sentLen)
            train_se += self.evaluate_on_batch(sess, train_x, train_y, mask, train_sentLen)
            prog.update(i + 1, [("train loss", loss)])

        train_obs = self.train_x.shape[0]
        train_mse = train_se / train_obs

        print 'Training MSE is {0}'.format(train_mse)

        print "Evaluating on dev set",
        dev_se = 0.0
        for i, (dev_x, dev_y, dev_sentLen, dev_mask) in enumerate(minibatches(self.dev_x, self.dev_y, self.dev_sentLen, self.dev_mask, self.config.batch_size)):
            dev_se += self.evaluate_on_batch(sess, dev_x, dev_y, dev_mask, dev_sentLen)

        dev_obs = self.dev_x.shape[0]
        dev_mse = dev_se / dev_obs

        print "- dev MSE: {:.2f}".format(dev_mse)
        return dev_mse

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

    # def __init__(self, encoder):
    #     self.train_x = encoder.train_x
    #     self.train_y = encoder.train_y
    #     self.train_mask = encoder.train_mask
    #     self.dev_x = encoder.dev_x
    #     self.dev_y = encoder.dev_y
    #     self.dev_mask = encoder.dev_mask
    #     self.pretrained_embeddings = encoder.pretrained_embeddings
    #     self.train_sentLen = encoder.train_sentLen
    #     self.dev_sentLen = encoder.dev_sentLen
    #     # Update our config with data parameters
    #     self.config = encoder.config
    #     self.config.max_sentence = encoder.config.max_sentence
    #     self.config.n_class = encoder.config.n_class
    #     self.config.embedding_size = encoder.config.embedding_size
    #     self.encoder = encoder
    #     self.maskId = encoder.maskId
    #     self.build()

'''
Read in Data
'''

train = '/Users/henryneeb/CS224N-Project/source/rcnn-master/beer/reviews.aspect1.small.train.txt.gz'
dev = '/Users/henryneeb/CS224N-Project/source/rcnn-master/beer/reviews.aspect1.small.heldout.txt.gz'
embedding = '/Users/henryneeb/CS224N-Project/source/rcnn-master/beer/review+wiki.filtered.200.txt.gz'

def main(debug=False):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    config = Config()
    ## this is where we add our own data
    # parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
    # if not os.path.exists('./data/weights/'):
    #     os.makedirs('./data/weights/')

    with tf.Graph().as_default():
        print "Building model...",
        start = time.time()
        ## this is where we add our model class name
        ## config is also a class name
        generatorModel = RNNGeneratorModel(config, embedding, train, dev)

        # generatorModel = RNNGeneratorModel(encoderModel)
        print "took {:.2f} seconds\n".format(time.time() - start)

        init = tf.global_variables_initializer()
        # If you are using an old version of TensorFlow, you may have to use
        # this initializer instead.
        # init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)

            # tvar = tf.trainable_variables()
            # for v in tvar:
            #     print v

            print 80 * "="
            print "TRAINING"
            print 80 * "="
            ## this is a function is the model class
            # model.run_epoch(session)
            generatorModel.fit(session, saver)
            # model.train_on_batch(session, model.train_x[0:32,:], model.train_y[0:32,:], model.train_mask[0:32,:])
            # train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch)
            # model.fit(session, None)
            # model.fit(session, saver)
            #
            # # train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch)
            # # model.fit(session, saver, parser, train_examples, dev_set)
            #
            # if not debug:
            #     print 80 * "="
            #     print "TESTING"
            #     print 80 * "="
            #     print "Restoring the best model weights found on the dev set"
            #     saver.restore(session, './parser.weights')
            #     print "Final evaluation on test set",
            #     ## we won't have this. we need function in our model that will evaluate on test set
            #     ## this is a function that will only calculate loss, "Evaluate function" takes inputs and compares to labels
            #     ## ie model.evaluate(test_set)
            #     loss = model.evaluate(test_set)
            #     print "- test UAS: {:.2f}".format(UAS * 100.0)
            #     print "Writing predictions"
            #     with open('q2_test.predicted.pkl', 'w') as f:
            #         cPickle.dump(dependencies, f, -1)
            #     print "Done!"


if __name__ == '__main__':
    main()
