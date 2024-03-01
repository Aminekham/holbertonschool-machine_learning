#!/usr/bin/env python3
"""
using a functional API
instead of the sequential one
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    the functional model gives
    the possibility of building more
    complex models by using the same layer
    over and over again and not always
    in the sequential order
    """
    input = K.Input(shape=(nx, ))
    x = input
    for i in range(len(activations)):
        x = K.layers.Dense(activation=activations[i],
                           units=layers[i],
                           kernel_regularizer=K.regularizers.l2(lambtha))(x)
        if i < len(activations) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
    model = K.Model(inputs=input,  outputs=x)
    return model
