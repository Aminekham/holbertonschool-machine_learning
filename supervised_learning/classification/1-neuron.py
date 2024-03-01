#!/usr/bin/env python3
"""
Creating the neural network by our
own to use it
"""


import numpy as np


class Neuron:
    """
    This is the neuron responsible for performing
    the classification task
    """
    __W = None
    __b = None
    __A = None

    def __init__(self, nx):
        """
        Initialization function to get the
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
    def W(self):
        """
        Weights getter
        """
        return self.__W

    @property
    def b(self):
        """
        Bias value getter
        """
        return self.__b

    @property
    def A(self):
        """
        A value getter
        """
        return self.__A
