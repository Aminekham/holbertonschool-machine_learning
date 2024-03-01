#!/usr/bin/env python3
"""
Using learning rate decay
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False,
                early_stopping=False, patience=0, decay_rate=1,
                alpha=0.1, learning_rate_decay=False):
    """
    working on learning rate decay
    by imprving it on every step and
    in proportion to the validation
    data
    """
    callbacks = []
    if early_stopping and validation_data:
        early = K.callbacks.EarlyStopping(monitor="val_loss",
                                          patience=patience,
                                          mode='min')
        callbacks.append(early)
    if learning_rate_decay and validation_data:
        def scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)
        lr_decay = K.callbacks.LearningRateScheduler(
            schedule=scheduler, verbose=1)
        callbacks.append(lr_decay)
    history = network.fit(data, labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=verbose,
                          callbacks=callbacks,
                          validation_data=validation_data,
                          shuffle=shuffle)
    return history