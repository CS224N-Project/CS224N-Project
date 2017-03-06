
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



# train = "/Users/alysonkane/Desktop/Classes/4 - Winter 2017/cs224n/project/beer/reviews.aspect1.small.train.txt"
# dev = "/Users/alysonkane/Desktop/Classes/4 - Winter 2017/cs224n/project/beer/reviews.aspect1.small.heldout.txt"
# embedding = "/Users/alysonkane/Desktop/Classes/4 - Winter 2017/cs224n/project/beer/review+wiki.filtered.200.txt.gz"
#
#
# ## read in data
# train_x, train_y, max_train = myio_read_annotations(train)
# dev_x, dev_y, max_dev = myio_read_annotations(dev)
# embedding_layer = myio_create_embedding_layer(embedding)
#
# ## maps words to int id
# train_x = [ embedding_layer.map_to_ids(x)[:max_train] for x in train_x ]
# dev_x = [ embedding_layer.map_to_ids(x)[:max_dev] for x in dev_x ]
#
# ## dictionary mapping int id to embedding
# embeddingDict = embedding_layer.embeddings
