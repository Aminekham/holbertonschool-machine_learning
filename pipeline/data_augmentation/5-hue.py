#!/usr/bin/env python3
import tensorflow as tf
"""
changing the hue
"""
def change_hue(image, delta):
    """
    changing the hue with a certain max control
    using the delta variable
    """
    return (tf.image.adjust_hue(image, delta))
