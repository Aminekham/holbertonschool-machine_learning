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
    __L = None
    __cache = None
    __weights = None

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

        self.__L = len(layers)
        self.__cache = dict()
        self.__weights = dict()
        for i in range(len(layers)):
            if type(layers[i]) != int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                He = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.__weights["W" + str(i + 1)] = He
            else:
                He = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
                self.__weights['W' + str(i + 1)] = He
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
        layers number getter
        """
        return self.__L

    @property
    def cache(self):
        """
        dnn cache getter
        """
        return self.__cache

    @property
    def weights(self):
        """
        weights getter
        """
        return self.__weights

    def forward_prop(self, X):
        """
        applying forward propagation in a layer
        using the sigmoid function and all the weights
        """
        self.cache['A0'] = X
        for l in range(1, self.L + 1):
            W = self.weights['W' + str(l)]
            b = self.weights['b' + str(l)]
            Z = np.dot(W, self.cache['A' + str(l - 1)]) + b
            A = 1 / (1 + np.exp(-Z))
            self.cache['A' + str(l)] = A
        return A, self.cache
