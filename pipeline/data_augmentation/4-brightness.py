#!/usr/bin/env python3
import tensorflow as tf
"""
adjusting the brightness
"""
def change_brightness(image, max_delta):
    """
    changing brightness randomly while controlling
    the maximum relative change with max delta
    """
    return (tf.image.random_brightness(image, max_delta))
