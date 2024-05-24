#!/usr/bin/env python3
"""

"""
import numpy as np
from tensorflow.keras.layers import Embedding

def gensim_to_keras(model):
    """
    
    """
    weights = model.wv.vectors
    vocab_size, embedding_dim = weights.shape
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[weights], trainable=True)    
    return embedding_layer
