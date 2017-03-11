class Config(object):
    drop_out = 0.5
    hidden_size = 200
    batch_size = 32
    epochs = 2
    lr = 0.001
    l2Reg = 1.0e-6
    # built when we construct the model
    max_sentence = 0
    n_class = 0
    embedding_size = 0