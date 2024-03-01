#!/usr/bin/env python3
"""
Normalizes the input array X using mean and standard deviation
"""


import numpy as np


def normalize(X, m, s):
    """
    the mean method normalization
    """
    for i in range(len(X[0])):
        for j in range(len(X)):
            standard = (X[j][i] - m[i]) / s[i]
            X[j][i] = standard
    return(X)
