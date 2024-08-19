#!/usr/bin/env python3
import tensorflow as tf
"""
This is a function to crop an image
"""

def crop_image(image, size):
    return tf.image.random_crop(image, size)
