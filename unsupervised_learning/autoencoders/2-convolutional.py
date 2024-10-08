#!/usr/bin/env python3
"""
Convolutional neural network
to do the encoding and decoding
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Compressing the images dimensions using a convolution
    encoder and doing the upsampling to get back again
    to the original images
    """
    encoder_inputs = keras.layers.Input(shape=input_dims)
    x = encoder_inputs
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.models.Model(encoder_inputs, x)
    decoder_inputs = keras.layers.Input(shape=latent_dims)
    x = decoder_inputs
    for f in reversed(filters):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(input_dims[-1], (3, 3),
                            activation='sigmoid', padding='same')(x)
    decoder_outputs = x
    decoder = keras.models.Model(decoder_inputs, decoder_outputs)
    autoencoder_outputs = decoder(encoder(encoder_inputs))
    autoencoder = keras.models.Model(encoder_inputs, autoencoder_outputs)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
