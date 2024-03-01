#!/usr/bin/env python3
"""
implementing early stopping
for our model using keras
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False,
                early_stopping=False, patience=0):
    """
    basing the callbacks on our early
    stopping if the validation data
    exists to avoid overfitting by
    monitoring each time a certain loss
    between the error on the training data
    and on validation data(new data points)
    callbacks in this context is the a way
    in keras to control the behaviour in
    a certain epoch based on some functions
    """
    callbacks = []
    if validation_data and early_stopping:
        early = K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        callbacks.append(early)
    trained = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)
    return trained
