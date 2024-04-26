#!/usr/bin/env python3
"""
building a vanilla
autoencoder: encoder and decoder
"""
import numpy as np
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    
    """
    input = tf.keras.Input(shape=(input_dims, ))
    layer = input
    for node in hidden_layers:
        layer = tf.keras.Dense(node, activation='relu')(layer)
    latent = tf.keras.Dense(latent_dims, activation='relu')(layer)
    encoder = tf.keras.models.Model(input, latent)
    layer = latent
    for node in reversed(hidden_layers):
        layer = tf.keras.Dense(node, activation='relu')(layer)
    decoder_layer = tf.keras.Dense(input_dims, activation='sigmoid')(layer)
    decoder = tf.keras.Model(latent, decoder_layer)
    autoencoder = tf.keras.Model(input, decoder_layer)
    return encoder, decoder, autoencoder