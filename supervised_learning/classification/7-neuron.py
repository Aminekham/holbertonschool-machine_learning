#!/usr/bin/env python3
"""
Creating the neural network by our
own
"""


import numpy as np
import matplotlib as plt


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

    def forward_prop(self, X):
        """
        calculating the z and the output of
        our softmax function for each
        input
        """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        calculating the loss and then
        calculate the whole cost of
        our neuron and return to
        reduce it in the next steps
        """
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
        prediction_ones = np.where(prediction >= 0.5, 1, 0)
        cost = self.cost(Y, prediction)
        return(prediction_ones, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = len(Y[0])
        dz = A - Y
        dw = 1 / m * np.dot(dz, X.T)
        db = 1 / m * np.sum(dz)
        self.__W -= alpha * dw
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """doing forward propagation for a certain number of iterations"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        costs = []
        for i in range(iterations):
            self.gradient_descent(X, Y, self.forward_prop(X), alpha)
            _, cost = self.evaluate(X, Y)
            if i % step == 0 or i == 0 or i == iterations:
                costs.append(cost)
            if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(range(0, iterations + 1, step), costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
