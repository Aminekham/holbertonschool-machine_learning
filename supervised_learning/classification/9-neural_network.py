#!/usr/bin/env python3
"""Module containing the NeuralNetwork class"""


import numpy as np


class NeuralNetwork:
    """
    NeuralNetwork class with one hidden layer performing
    binary classification
    """

    __W1 = None
    __b1 = None
    __A1 = None
    __W2 = None
    __b2 = None
    __A2 = None

    def __init__(self, nx, nodes):
        """Initializer method with constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for weights vector for the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Getter for bias for the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Getter for activated output for the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Getter for weights vector for the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Getter for bias for the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Getter for activated output for the output neuron (prediction)"""
        return self.__A2
