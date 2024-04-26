#!/usr/bin/env python3
"""
building a vanilla
autoencoder: encoder and decoder
"""
import tensorflow.keras as keras

def autoencoder(input_dims, hidden_layers, latent_dims):
    """

    """
    input = keras.Input(shape=(input_dims, ))
    layer = input
    for node in hidden_layers:
        layer = keras.layers.Dense(node, activation='relu')(layer)
    latent = keras.layers.Dense(latent_dims, activation='relu')(layer)
    encoder = keras.models.Model(input, latent)
    layer = latent
    for node in reversed(hidden_layers):
        layer = keras.layers.Dense(node, activation='relu')(layer)
    decoder_layer = keras.layers.Dense(input_dims, activation='sigmoid')(layer)
    decoder = keras.Model(latent, decoder_layer)
    autoencoder_outputs = decoder(encoder(input))
    autoencoder = keras.Model(input, autoencoder_outputs)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, autoencoder
