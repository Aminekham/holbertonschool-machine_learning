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

    def forward_prop(self, X):
        """using the sigmoid function for all the neurons"""
        Z1 = np.dot(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """calculating the cost of a certain neuron"""
        sub = 1.0000001 - A
        cost = -1/len(Y[0]) * np.sum(Y * np.log(A) + (1 - Y) * np.log(sub))
        return cost

    def evaluate(self, X, Y):
        """
        calculating the loss and then
        calculate the whole cost of
        our neuron and return to
        reduce it in the next steps
        """
        prediction = self.forward_prop(X)
        prediction_ones = np.where(prediction[1] >= 0.5, 1, 0)
        cost = self.cost(Y, prediction[1])
        return(prediction_ones, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        gradient descent
        """
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = (1 / m) * np.dot(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2)
        dz1 = np.dot(self.W2.T, dz2) * A1 * (1 - A1)
        dw1 = (1 / m) * np.dot(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1)
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2
