#!/usr/bin/env python3
"""
Calculating the constants needed for data
normalization process
"""


import numpy as np


def normalization_constants(X):
    """
    calculating the mean and
    standard deviation
    """
    mean = []
    std_variation = []
    for i in range(len(X[0])):
        this_sum = 0
        for j in range(len(X)):
            this_sum = this_sum + X[j][i]
        mean.append(round(this_sum / len(X), 8))

    for i in range(len(X[0])):
        this_sum = 0
        for j in range(len(X)):
            this_sum = this_sum + (X[j][i] - mean[i])**2
        std_variation.append(round((this_sum / len(X))**0.5, 8))
    mean = np.array(mean)
    std_variation = np.array(std_variation)
    return(mean, std_variation)
