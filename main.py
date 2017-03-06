import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from preprocess import myio_create_embedding_layer, myio_read_annotations


### RNN Cell
class RNNCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our RNN cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size, name_suffix):
        self.input_size = input_size
        self._state_size = state_size
        self._name_suffix = name_suffix

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        """Updates the state using the previous @state and @inputs.
        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            ### YOUR CODE HERE (~6-10 lines)

            ## layer one 
            W_x = tf.get_variable(name = "W_x" + str(self._name_suffix),
                                  shape = (self.input_size, self._state_size),
                                  dtype = tf.float32,
                                  initializer = tf.contrib.layers.xavier_initializer())

            W_h = tf.get_variable(name = "W_h" + str(self._name_suffix),
                                  shape = (self._state_size, self._state_size),
                                  dtype = tf.float32,
                                  initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name = "b" + str(self._name_suffix),
                                shape = (self._state_size,),
                                dtype = tf.float32,
                                initializer = tf.constant_initializer(0.0))


            output = tf.tanh(tf.matmul(inputs, W_x) + tf.matmul(state, W_h) + b)
            ### END YOUR CODE ###
        return output


def data_iterator(data, labels, batch_size, sentLen):
    """ A simple data iterator """
    numObs = data.shape[0]
    while True:
        # shuffle labels and features
        idxs = np.arange(0, numObs)
        np.random.shuffle(idxs)
        shuffledData = data[idxs]
        shuffledLabels = labels[idxs]
        shuffledSentLen = sentLen[idxs]
        for idx in range(0, numObs, batch_size):
            dataBatch = shuffledData[idx:idx + batch_size]
            labelsBatch = shuffledLabels[idx:idx + batch_size]
            seqLenBatch = shuffledSentLen[idx:idx + batch_size]
            yield dataBatch, labelsBatch, seqLenBatch

'''
Read in Data
'''

train = '/Users/stanford/Desktop/Winter2017/CS224n/FinalProject/beer/reviews.aspect1.small.train.txt.gz'
dev = '/Users/stanford/Desktop/Winter2017/CS224n/FinalProject/beer/reviews.aspect1.small.heldout.txt.gz'
embedding = '/Users/stanford/Desktop/Winter2017/CS224n/FinalProject/beer/review+wiki.filtered.200.txt.gz'


## read in data
train_x, train_y, max_train = myio_read_annotations(train)
train_y = np.array(train_y)
dev_x, dev_y, max_dev = myio_read_annotations(dev)
embedding_layer = myio_create_embedding_layer(embedding)

## maps words to int id
train_x = [ embedding_layer.map_to_ids(x)[:max_train] for x in train_x ]
dev_x = [ embedding_layer.map_to_ids(x)[:max_dev] for x in dev_x ]

## dictionary mapping int id to embedding
embeddingDict = embedding_layer.embeddings

'''
Data Parameters
'''

# define modelling parameters
# 804
max_sentence = 915
n_class = train_y.shape[1]
embedding_size = embeddingDict.shape[1]
drop_out = 0.5
hidden_size = 200
batch_size = 32
epochs = 10
lr = 0.001
l2Reg = 1.0e-6

'''
Mask Data
'''

maskId = embeddingDict.shape[0]
sentLen = np.array(map(len, train_x), dtype = np.int32)
sentDiff = list(max_sentence - sentLen)
paddings = [np.full(shape = x, fill_value = maskId, dtype=np.int32) for x in sentDiff]
zipped = zip(train_x, paddings)
train_x_pad = [np.append(x[0], x[1]) for x in zipped]
train_x_pad = np.array(train_x_pad)
# train_x_pad = list(train_x_pad)
# train_x_pad = map(list, train_x_pad)
# train_x_pad = [[[x] for x in l] for l in train_x_pad]
# train_x_pad = [map(int, l) for l in train_x_pad]
# train_x_pad = [map(list, l) for l in train_x_pad]

# modify embeddings to get zero paddings
paddEmbed = np.zeros(shape = (1, embedding_size), dtype = np.float32)
embeddingDictPad = np.append(embeddingDict, paddEmbed, axis = 0)

'''
Placeholders
'''

# batchSize X sentence X numClasses
inputPH = tf.placeholder(dtype = tf.int32,
                         shape = (batch_size, max_sentence),
                         name = 'input')
# batchSize X numClasses
labelsPH = tf.placeholder(dtype = tf.int32,
                          shape = (batch_size, n_class),
                          name = 'labels')
# mask over sentences not long enough
maskPH = tf.placeholder(dtype = tf.bool,
                        shape = (batch_size, max_sentence),
                        name = 'mask')
dropoutPH = tf.placeholder(dtype = tf.float32,
                           shape = (),
                           name = 'dropout')
seqPH = tf.placeholder(dtype = tf.float32,
                       shape = (batch_size,),
                       name = 'sequenceLen')
l2RegPH = tf.placeholder(dtype = tf.float32,
                         shape = (),
                         name = 'l2Reg')

'''
Get Embeddings
'''
embeddings = tf.constant(embeddingDictPad, dtype = tf.float32)
embedInput = tf.nn.embedding_lookup(embeddings, inputPH)
embedInput = tf.reshape(embedInput,
                        shape = (batch_size, max_sentence, embedding_size))
# revEmbedInput = tf.reverse(embedInput, dims = [False, True, False])
# embedInput = tf.unstack(embedInput, axis = 1)
revEmbedInput = embedInput[::-1]

'''
Batch our Data
'''
iter = data_iterator(train_x_pad, train_y, batch_size, sentLen)

'''
Model Declaration
'''

###########
# ENCODER #
###########


preds = [] # predicted output at each timestep

cell1 = RNNCell(hidden_size, hidden_size, "cell1")
cell2 = RNNCell(hidden_size, hidden_size, "cell2")

# Extract sizes
nHid = hidden_size
nClass = n_class

W = tf.get_variable(name = 'W',
                    shape = ((2*nHid), nClass),
                    dtype = tf.float32,
                    initializer = tf.contrib.layers.xavier_initializer())

b =tf.get_variable(name = 'b',
                    shape = (nClass,),
                    dtype = tf.float32,
                    initializer = tf.constant_initializer(0.0))

h1_Prev = tf.zeros(shape = (tf.shape(embedInput)[0], nHid), dtype = tf.float32)
h2_Prev = tf.zeros(shape = (tf.shape(embedInput)[0], nHid), dtype = tf.float32)

for time_step in range(max_sentence):
    if time_step > 0:
        tf.get_variable_scope().reuse_variables()

    # First RNN Layer - uses embeddings
    h1_t = cell1(embedInput[:, time_step, :], h1_Prev)
    h1_drop_t = tf.nn.dropout(h1_t, keep_prob = dropoutPH)

    # Second RNN Layer - uses First layer hidden states
    h2_t = cell2(h1_drop_t, h2_Prev)
    h2_drop_t = tf.nn.dropout(h2_t, keep_prob = dropoutPH)

    h1_Prev = h1_t
    h2_Prev = h2_t


# Concatenate last states of first and second layer for prediction layer
h_t = tf.concat(concat_dim = 1, values = [h1_drop_t, h2_drop_t])
y_t = tf.tanh(tf.matmul(h_t, W) + b)
preds.append(y_t)

# Compute L2 loss
loss = tf.nn.l2_loss(tf.cast(labelsPH, dtype=tf.float32) - preds)
loss = tf.reduce_mean(loss)

# # Apply L2 regularization
# regularization = tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

# totalCost = (10.0 * loss) + (l2RegPH * regularization)

# opt = tf.train.AdamOptimizer(learning_rate = lr)
# train_op = opt.minimize(totalCost)



##################
# GENERATOR STEP #
##################

# output2Rev = tf.reverse(output2, axis = 1)

# hFinal = tf.concat(concat_dim = 0, values = [output1, output2Rev])

'''
Debug
'''
# blah = tf.nn.embedding_lookup(embeddings, train_x_pad[0:32])
# blah2 = tf.train.batch([blah, train_y[0:32]], 32)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for _ in range(1):
        x, y, seqLen = iter.next()
        mask = ((x != maskId) + 0.0)
        print sess.run(loss, feed_dict = {
            inputPH: x,
            labelsPH: y,
            maskPH: mask,
            dropoutPH: drop_out,
            seqPH: seqLen,
            l2RegPH: l2Reg
        })
        # print output1
        # print state1
        # print type(state1)
        # print type(output1)
        # print tf.shape(state1)
        # print [v for v in tf.global_variables()]
        # print type(embedInput)
        # print len(embedInput)
        # print tf.shape(embedInput[0])
        # print embedInput[0]
    # print (sess.run(blah))
    # print (sess.run(tf.shape(blah)))

exit()

'''
Feed Dictionary
'''

feed_dict = {
    inputPH: inputs_batch,
    self.mask_placeholder: mask_batch,
    self.dropout_placeholder: dropout
}

# Add labels if not none
if labels_batch is not None:
    feed_dict[self.labels_placeholder] = labels_batch


with tf.Graph().as_default():



    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)
        model.fit(session, saver, train, dev)
        if report:
            report.log_output(model.output(session, dev_raw))
            report.save()
        else:
            # Save predictions in a text file.
            output = model.output(session, dev_raw)
            sentences, labels, predictions = zip(*output)
            predictions = [[LBLS[l] for l in preds] for preds in predictions]
            output = zip(sentences, labels, predictions)

            with open(model.config.conll_output, 'w') as f:
                write_conll(f, output)
            with open(model.config.eval_output, 'w') as f:
                for sentence, labels, predictions in output:
                    print_sentence(f, sentence, labels, predictions)