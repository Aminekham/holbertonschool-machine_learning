#!/usr/bin/env python3
"""
Full cnn using keras
"""

import tensorflow.keras as K


def lenet5(X):
    """

    """
    model = K.Sequential()
    model.add(K.layers.Conv2D(input_shape = (X.shape[1:]),filters=6, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(K.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='valid'))
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(units=120, activation='relu'))
    model.add(K.layers.Dense(units=84, activation='relu'))
    model.add(K.layers.Dense(units=10, activation='softmax'))
    model.compile(optimizer='Adam', metrics=['accuracy'], loss='categorical_crossentropy')
    return model
