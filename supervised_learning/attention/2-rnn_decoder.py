import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dense
SelfAttention = __import__('1-self_attention').SelfAttention
"""

"""


class RNNDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
        self.units = units
        self.batch = batch
        self.embedding = Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = GRU(units, return_sequences=True, return_state=True,
                       recurrent_initializer='glorot_uniform')
        self.F = Dense(vocab)
        self.attention = SelfAttention(units)
    
    def call(self, x, s_prev, hidden_states):
        x = self.embedding(x)
        context, _ = self.attention(s_prev, hidden_states)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        output, s = self.gru(x, initial_state=s_prev)
        y = self.F(tf.reshape(output, (-1, output.shape[2])))
        return y, s
