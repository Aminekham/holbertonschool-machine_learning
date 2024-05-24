#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU
from tensorflow.keras.initializers import glorot_uniform
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
        self.embedding = Embedding(vocab, embedding)
        self.gru = GRU(units,
                       recurrent_initializer=glorot_uniform(),
                       return_sequences=True,
                       return_state=True)

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
