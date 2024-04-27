#!/usr/bin/env python3
"""

"""
import tensorflow.keras as keras

def autoencoder(input_dims, filters, latent_dims):
    encoder_inputs = keras.layers.Input(shape=input_dims)
    x = encoder_inputs
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.models.Model(encoder_inputs, x, name='encoder')
    decoder_inputs = keras.layers.Input(shape=latent_dims)
    x = decoder_inputs
    for f in reversed(filters):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    decoder = keras.models.Model(decoder_inputs, x, name='decoder')
    autoencoder_outputs = decoder(encoder(encoder_inputs))
    autoencoder = keras.models.Model(encoder_inputs, autoencoder_outputs, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
