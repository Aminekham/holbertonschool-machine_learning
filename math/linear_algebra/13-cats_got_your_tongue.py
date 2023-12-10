#!/usr/bin/env python3
"""
method to concatenate in numpy
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    using np.concatenate for matrixs
    """
    return(np.concatenate([mat1, mat2], axis=axis))
