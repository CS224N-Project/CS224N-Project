
import os, sys, gzip
import time
import math
import json
import cPickle as pickle

import numpy as np
import random
import argparse
from collections import OrderedDict

class EmbeddingLayer(object):
    '''
        Embedding layer that
                (1) maps string tokens into integer IDs
                (2) maps integer IDs into embedding vectors (as matrix)

        Inputs
        ------

        n_d             : dimension of word embeddings; may be over-written if embs
                            is specified
        vocab           : an iterator of string tokens; the layer will allocate an ID
                            and a vector for each token in it
        oov             : out-of-vocabulary token
        embs            : an iterator of (word, vector) pairs; these will be added to
                            the layer
        fix_init_embs   : whether to fix the initial word vectors loaded from embs

    '''
    def __init__(self, n_d, vocab, oov="<unk>", embs=None, fix_init_embs=True):

        if embs is not None:
            lst_words = [ ]
            vocab_map = {}
            emb_vals = [ ]
            for word, vector in embs:
                assert word not in vocab_map, "Duplicate words in initial embeddings"
                vocab_map[word] = len(vocab_map)
                emb_vals.append(vector)
                lst_words.append(word)

            self.init_end = len(emb_vals) if fix_init_embs else -1
            if n_d != len(emb_vals[0]):
                say("WARNING: n_d ({}) != init word vector size ({}). Use {} instead.\n".format(
                        n_d, len(emb_vals[0]), len(emb_vals[0])
                    ))
                n_d = len(emb_vals[0])

            say("{} pre-trained embeddings loaded.\n".format(len(emb_vals)))

            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    emb_vals.append(random_init((n_d,))*(0.001 if word != oov else 0.0))
                    lst_words.append(word)

            #emb_vals = np.vstack(emb_vals).astype(theano.config.floatX)
            emb_vals = np.vstack(emb_vals).astype(float)
            self.vocab_map = vocab_map
            self.lst_words = lst_words
        else:
            lst_words = [ ]
            vocab_map = {}
            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    lst_words.append(word)

            self.lst_words = lst_words
            self.vocab_map = vocab_map
            emb_vals = random_init((len(self.vocab_map), n_d))
            self.init_end = -1

        if oov is not None and oov is not False:
            assert oov in self.vocab_map, "oov {} not in vocab".format(oov)
            self.oov_tok = oov
            self.oov_id = self.vocab_map[oov]
        else:
            self.oov_tok = None
            self.oov_id = -1

        #self.embeddings = create_shared(emb_vals)
        self.embeddings = emb_vals
        if self.init_end > -1:
            self.embeddings_trainable = self.embeddings[self.init_end:]
        else:
            self.embeddings_trainable = self.embeddings

        self.n_V = len(self.vocab_map)
        self.n_d = n_d

    def map_to_words(self, ids):
        n_V, lst_words = self.n_V, self.lst_words
        return [ lst_words[i] if i < n_V else "<err>" for i in ids ]

    def map_to_ids(self, words, filter_oov=False):
        '''
            map the list of string tokens into a numpy array of integer IDs

            Inputs
            ------

            words           : the list of string tokens
            filter_oov      : whether to remove oov tokens in the returned array


            Outputs
            -------

            return the numpy array of word IDs

        '''
        vocab_map = self.vocab_map
        oov_id = self.oov_id
        if filter_oov:
            not_oov = lambda x: x!=oov_id
            return np.array(
                    filter(not_oov, [ vocab_map.get(x, oov_id) for x in words ]),
                    dtype="int32"
                )
        else:
            return np.array(
                    [ vocab_map.get(x, oov_id) for x in words ],
                    dtype="int32"
                )

    def forward(self, x):
        '''
            Fetch and return the word embeddings given word IDs x

            Inputs
            ------

            x           : a theano array of integer IDs


            Outputs
            -------

            a theano matrix of word embeddings
        '''
        return self.embeddings[x]

    @property
    def params(self):
        return [ self.embeddings_trainable ]

    @params.setter
    def params(self, param_list):
        self.embeddings.set_value(param_list[0].get_value())

def say(s, stream=sys.stdout):
    stream.write("{}".format(s))
    stream.flush()

default_rng = np.random.RandomState(random.randint(0,9999))
def random_init(size, rng=None, rng_type=None):
    if rng is None: rng = default_rng
    if rng_type is None:
        #vals = rng.standard_normal(size)
        vals = rng.uniform(low=-0.05, high=0.05, size=size)

    elif rng_type == "normal":
        vals = rng.standard_normal(size)

    elif rng_type == "uniform":
        vals = rng.uniform(low=-3.0**0.5, high=3.0**0.5, size=size)

    else:
        raise Exception(
            "unknown random inittype: {}".format(rng_type)
          )

    return vals.astype(float)


def myio_read_annotations(path):
    data_x, data_y = [ ], [ ]
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            y, sep, x = line.partition("\t")
            x, y = x.split(), y.split()
            if len(x) == 0: continue
            y = np.asarray([ float(v) for v in y ])
            data_x.append(x)
            data_y.append(y)
    say("{} examples loaded from {}\n".format(
            len(data_x), path
        ))
    say("max text length: {}\n".format(
        max(len(x) for x in data_x)
    ))
    return data_x, data_y, max(len(x) for x in data_x)


def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                vals = np.array([ float(x) for x in parts[1:] ])
                yield word, vals

### /utils/__init__.py END ###

### myio.py BEGIN ###

def myio_create_embedding_layer(path):
    embedding_layer = EmbeddingLayer(
            n_d = 200,
            vocab = [ "<unk>", "<padding>" ],
            embs = load_embedding_iterator(path),
            oov = "<unk>",
            #fix_init_embs = True
            fix_init_embs = False
        )
    return embedding_layer

def padData(data, embeddingDict):
    '''
    Adds padding ids to fill our training data so all reviews are the same size
    :param data: training data with the tokenized reviews
    :param embeddingDict: the embedding dictionary with pretrained vectors
    :return: padded data and embedding with padding word vector
    '''
    maskId = embeddingDict.shape[0]

    # Find the max sentence length
    # maxLength = np.max(map(len, data))
    maxLength = 1145

    # Get the length of each sentence
    sentLen = np.array(map(len, data), dtype=np.int32)
    sentDiff = maxLength - sentLen

    # Fill each sentence to max sentence length
    paddings = [np.full(shape=x, fill_value=maskId, dtype=np.int32) for x in
                sentDiff]
    sentAndPad = zip(data, paddings)
    dataPad = [np.append(x[0], x[1]) for x in sentAndPad]
    dataPad = np.array(dataPad)

    # Add extra padding vector to embeddings
    embedding_size = embeddingDict.shape[1]
    paddEmbed = np.zeros(shape=(1, embedding_size), dtype=np.float32)
    embeddingDictPad = np.append(embeddingDict, paddEmbed, axis=0)

    # create mask for data
    mask = (dataPad != maskId)

    return dataPad, embeddingDictPad, mask, sentLen

def readOurData(trainPath, devPath, testPath, embeddingPath):
    '''
    Wrapper function that reads in padded training and development data
    :param trainPath: path to training data
    :param devPath: path to development data
    :param embeddingPath: path to embedding dictionary
    :return: padded training and development data (reviews and labels).
    Embedding dictionary with a padding vector. Also returns masks for training
    and development data
    '''

    # Read in embeddings
    embedding_layer = myio_create_embedding_layer(embeddingPath)
    embeddingDict = embedding_layer.embeddings

    # Read in training data
    train_x, train_y, max_train = myio_read_annotations(trainPath)
    train_y = np.array(train_y)
    train_x = [embedding_layer.map_to_ids(x)[:max_train] for x in train_x]

    # Read in development data
    dev_x, dev_y, max_dev = myio_read_annotations(devPath)
    dev_y = np.array(dev_y)
    dev_x = [embedding_layer.map_to_ids(x)[:max_dev] for x in dev_x]

    # Read in test data
    test_x, test_y, max_test = myio_read_annotations(devPath)
    test_y = np.array(test_y)
    test_x = [embedding_layer.map_to_ids(x)[:max_test] for x in test_x]

    # pad trainging and devlopment data
    train_x_pad, embedding_pad, train_mask, train_sentLen = padData(train_x, embeddingDict)
    dev_x_pad, _, dev_mask, dev_sentLen = padData(dev_x, embeddingDict)
    test_x_pad, _, test_mask, test_sentLen = padData(dev_x, embeddingDict)

    return train_x_pad, train_y, train_mask, train_sentLen, dev_x_pad, dev_y, dev_mask, dev_sentLen, embedding_pad, test_x_pad, test_y, test_mask, test_sentLen