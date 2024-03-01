#!/usr/bin/env python3
"""
building the core block of the
residual neural network(projection not residual)
"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    explanation later goes here
    """
    shortcut = K.layers.Conv2D(filters[2], kernel_size=(1, 1),
                               strides=(s, s), padding='same',
                               kernel_initializer=
                               K.initializers.he_normal(seed=
                                                        None))(A_prev)
    shortcut_bn = K.layers.BatchNormalization()(shortcut)
    first_conv = K.layers.Conv2D(filters[0], kernel_size=(1, 1),
                                 strides=(s, s), padding='same',
                                 kernel_initializer=
                                 K.initializers.he_normal(seed=
                                                          None))(A_prev)
    first_bn = K.layers.BatchNormalization()(first_conv)
    activation1 = K.layers.Activation("relu")(first_bn)
    second_conv = K.layers.Conv2D(filters[1], kernel_size=(3, 3),
                                  strides=(1, 1), padding='same',
                                  kernel_initializer=
                                  K.initializers.he_normal(seed=
                                                           None))(activation1)
    second_bn = K.layers.BatchNormalization()(second_conv)
    activation2 = K.layers.Activation("relu")(second_bn)
    third_conv = K.layers.Conv2D(filters[2], kernel_size=(1, 1),
                                 strides=(1, 1), padding='same',
                                 kernel_initializer=K.initializers.he_normal(seed=None))(activation2)
    third_bn = K.layers.BatchNormalization()(third_conv)
    activation3 = K.layers.Activation("relu")(third_bn)
    activated_output = K.layers.Add()([shortcut_bn, activation3])
    return activated_output
