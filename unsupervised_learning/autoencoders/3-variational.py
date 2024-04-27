#!/usr/bin/env python3
"""

"""
import tensorflow as tf
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """

    """
    encoder_inputs = keras.layers.Input(shape=(input_dims,))
    x = encoder_inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)

    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = keras.models.Model(encoder_inputs,
                                 [z, z_mean, z_log_var], name='encoder')
    decoder_inputs = keras.layers.Input(shape=(latent_dims,))
    x = decoder_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.models.Model(decoder_inputs,
                                 decoder_outputs, name='decoder')
    autoencoder_outputs = decoder(encoder(encoder_inputs)[0])
    autoencoder = keras.models.Model(encoder_inputs,
                                     autoencoder_outputs, name='autoencoder')
    reconstruction_loss = tf.keras.losses.binary_crossentropy(
        encoder_inputs, autoencoder_outputs)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    autoencoder.add_loss(vae_loss)
    autoencoder.compile(optimizer='adam')
    return encoder, decoder, autoencoder
