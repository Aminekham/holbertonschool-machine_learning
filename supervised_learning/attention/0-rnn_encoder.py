#!/usr/bin/env python3
import tensorflow as tf
"""

"""


class RNNEncoder(tf.keras.layers.Layer):
    """

    """
    def __init__(self, vocab, embedding, units, batch):
        """

        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        init = tf.keras.initializers.glorot_uniform()
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer=init,
                                       return_sequences=True, return_state=True)

    def initialize_hidden_state(self):
        """

        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """

        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
