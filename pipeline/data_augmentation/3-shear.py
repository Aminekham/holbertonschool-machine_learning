#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.preprocessing.image import apply_affine_transform
"""
This is a function to shear an image
"""

def shear_image(image, intensity):
    image = tf.keras.preprocessing.image.img_to_array(image)
    sheared_image = apply_affine_transform(image, shear=intensity)
    return tf.convert_to_tensor(sheared_image)
