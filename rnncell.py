import tensorflow as tf

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
        return output