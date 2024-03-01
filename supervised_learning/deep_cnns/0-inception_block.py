#!/usr/bin/env python3
"""
building the architecture
needed to match the idea of the
inception blocks
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    the presented artechture in the figure was
    introduced by me in 3 big steps which have the idea
    of using 1x1, 3x3 and 5x5 filters at the same time
    and not to choose one of them 
    first stack C1 C12 ... : The 1 layers and those the first ones to first project the
    input image to lower dimension using the (1, 1) convolutions
    to reduce the number of parameters and reduce the computations
    of the model
    and also using the maxpooling layer for the (1, 1) convultion after it
    second stack C2 C21 ...: using the (3, 3) convulution, (5, 5) convulution and (1, 1)
    convulution at the same time
    third stack inception: concatinating the output of the second stack
    """
    C1 = K.layers.Conv2D(filters=filters[0], kernel_size=(1, 1), padding='same',activation='relu')(A_prev)
    C11 = K.layers.Conv2D(filters=filters[1], kernel_size=(1, 1), padding='same', activation='relu')(A_prev)
    C12 = K.layers.Conv2D(filters=filters[3], kernel_size=(1, 1), padding='same',activation='relu')(A_prev)
    P13 = K.layers.MaxPooling2D(pool_size=(3, 3),strides=(1, 1), padding='same')(A_prev)
    C2 = K.layers.Conv2D(filters=filters[2], kernel_size=(3, 3), padding='same', activation='relu')(C11)
    C21 = K.layers.Conv2D(filters=filters[4], kernel_size=(5, 5), padding='same',activation='relu')(C12)
    C22 = K.layers.Conv2D(filters=filters[5], kernel_size=(1, 1), padding='same',activation='relu')(P13)
    inception = K.layers.Concatenate(axis=-1)([C1, C2, C21, C22])
    return inception
