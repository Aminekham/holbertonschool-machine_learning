#!/usr/bin/env python3
import tensorflow as tf
"""

"""


def sdp_attention(Q, K, V, mask=None):
    """
    Applying the attention equation where
    Q is the query(the whole sentence)
    K is the key(the compared element to v)
    V is the value(the element to be compared to K)
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)
    return output, attention_weights
