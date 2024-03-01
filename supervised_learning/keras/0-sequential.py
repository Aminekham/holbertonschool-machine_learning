#!/usr/bin/env python3
"""
Creating the sequential model
using keras
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Building the initial sequential
    model for deep learning
    """
    model = K.Sequential()
    for i in range(len(activations)):
        if i == 0:
            model.add(K.layers.Dense(input_shape=(nx, ),
                                     units=layers[i],
                                     activation=activations[0],
                                     kernel_regularizer=K.regularizers.l2(lambtha)))
        else:
            model.add(K.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=K.regularizers.l2(lambtha)))
        if i <  len(activations) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
