#!/usr/bin/env python3
import tensorflow as tf
"""
This is a function to flip an image
"""

def flip_image(image):
    """
    Applying the flipping image
    augmentation using tenserflow
    """
    return tf.image.flip_left_right(image)
