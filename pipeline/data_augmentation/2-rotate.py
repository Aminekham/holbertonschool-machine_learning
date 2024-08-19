#!/usr/bin/env python3
import tensorflow as tf
"""
This is a function to rotate an image
"""

def rotate_image(image):
    return tf.image.rot90(image, k=1)
