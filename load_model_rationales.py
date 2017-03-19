
import tensorflow as tf
from generator import RNNGeneratorModel
from config import Config
import time

# train = '/home/neuron/beer/reviews.aspect1.train.txt.gz'
# dev = '/home/neuron/beer/reviews.aspect1.heldout.txt.gz'
# embedding = '/home/neuron/beer/review+wiki.filtered.200.txt.gz'
# test = '/home/neuron/beer/annotations.txt.gz'
# annotations = '/home/neuron/beer/annotations.json'

train = '/Users/henryneeb/CS224N-Project/source/rcnn-master/beer/reviews.aspect1.small.train.txt.gz'
dev = '/Users/henryneeb/CS224N-Project/source/rcnn-master/beer/reviews.aspect1.small.heldout.txt.gz'
embedding = '/Users/henryneeb/CS224N-Project/source/rcnn-master/beer/review+wiki.filtered.200.txt.gz'
test = '/Users/henryneeb/CS224N-Project/source/rcnn-master/beer/annotations.txt.gz'
annotations = '/Users/henryneeb/CS224N-Project/source/rcnn-master/beer/annotations.json'

outFile = 'generator-RNN-testpreds.txt'

config = Config()

with tf.Graph().as_default():
    print "Building model...",
    start = time.time()
    # Construct a raw model
    model = RNNGeneratorModel(config, embedding, train, dev, test,
                                       annotations)
    print "took {:.2f} seconds\n".format(time.time() - start)
    # init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        # session.run(init)
        saver.restore(session, './generator.weights')
        model.save_preds(session, outFile)
        print 'Finished predictions'


