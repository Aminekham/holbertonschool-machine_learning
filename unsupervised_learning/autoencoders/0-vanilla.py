#!/usr/bin/env python3
"""
Basic vanilla autoencoder 
encoder and decoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """

    """
    X_input = keras.Input(shape=(input_dims,))
    hidden_ly = keras.layers.Dense(units=hidden_layers[0], activation='relu')
    Y_prev = hidden_ly(X_input)
    for i in range(1, len(hidden_layers)):
        hidden_ly = keras.layers.Dense(units=hidden_layers[i],
                                       activation='relu')
        Y_prev = hidden_ly(Y_prev)
    latent_ly = keras.layers.Dense(units=latent_dims, activation='relu')
    bottleneck = latent_ly(Y_prev)
    encoder = keras.Model(X_input, bottleneck)
    X_input_decoded = keras.Input(shape=(latent_dims,))
    hidden_ly_decoded = keras.layers.Dense(units=hidden_layers[-1],
                                           activation='relu')
    Y_prev = hidden_ly_decoded(X_input_decoded)
    for j in reversed(hidden_layers):
        hidden_ly_decoded = keras.layers.Dense(units=j,
                                               activation='relu')
        Y_prev = hidden_ly_decoded(Y_prev)
    last_layer = keras.layers.Dense(units=input_dims, activation='sigmoid')
    output = last_layer(Y_prev)
    decoder = keras.Model(X_input_decoded, output)
    encoder_o = encoder(X_input)
    decoder_o = decoder(encoder_o)
    auto = keras.Model(X_input, decoder_o)
    auto.compile(loss='binary_crossentropy', optimizer='adam')
    return encoder, decoder, auto
