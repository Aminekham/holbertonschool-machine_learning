#!/usr/bin/env python3
"""

"""
import tensorflow as tf


def create_masks(inputs, target):
    """

    """
    def create_padding_mask(seq):
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask
    encoder_mask = create_padding_mask(inputs)
    decoder_padding_mask = create_padding_mask(inputs)
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return encoder_mask, combined_mask, decoder_padding_mask
