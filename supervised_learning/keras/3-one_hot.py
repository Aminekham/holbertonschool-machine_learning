#!/usr/bin/env python3
"""
converting a normal
vector of labels
into a classification
matrix(one_hot matrix)
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    returning the one hot matrix
    needed for classification
    """
    hot_matrix = K.utils.to_categorical(labels, classes)
    return hot_matrix
