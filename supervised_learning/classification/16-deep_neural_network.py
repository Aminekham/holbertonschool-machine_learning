#!/usr/bin/env python3
"""
Creating the deep neural network
by my own
"""


import numpy as np


class DeepNeuralNetwork:
    """
    this is a deep neural network
    performing binary classification
    """
    def __init__(self, nx, layers):
        """
        initialisation function to get the number
        of layers to initialize the cache and
        initliaze the weights and biases
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = dict()
        self.weights = dict()
        for i in range(len(layers)):
            if type(layers[i]) != int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                He = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.weights["W" + str(i + 1)] = He
            else:
                He = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
                self.weights['W' + str(i + 1)] = He
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
