#!/usr/bin/env python3
import numpy as np
"""
Creating the neural network by our
own
"""


class Neuron:
    """
    this is the neuron responsible for performing
    the classification task
    """
    def __init__(self, nx):
        """
        initialisation function to get the
        weights entering the neuron
        while defining the bias b and the output A
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def weights_getter(self):
        """
        weights getter
        """
        return self.__W

    @property
    def b_getter(self):
        """
        bias value getter
        """
        return self.__b

    @property
    def A_getter(self):
        """
        A value getter
        """
        return self.__A
    
    def forward_prop(self, X):
        """
        calculating the z and the output of 
        our softmax function for each 
        input
        """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A
