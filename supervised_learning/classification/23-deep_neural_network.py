#!/usr/bin/env python3
"""
"""
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """
    """

    def __init__(self, nx, layers):
        """
        """
        if type(nx) is not int:
            raise TypeError('nx must be a positive integer')
        if nx < 1:
            raise ValueError('nx must be positive')
        if type(layers) is not list:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        last = nx
        for l in range(1, self.__L + 1):
            nodes = layers[l - 1]
            if type(nodes) is not int or nodes <= 0:
                raise TypeError('layers must be a list of positive integers')
            self.__weights['W' + str(l)] = (np.random.randn(nodes, last)
                                            * np.sqrt(2 / last))
            self.__weights['b' + str(l)] = np.zeros((nodes, 1))
            last = nodes

    @property
    def weights(self):
        return self.__weights

    @property
    def cache(self):
        return self.__cache

    @property
    def L(self):
        return self.__L

    def forward_prop(self, X):
        """
        """
        self.__cache = {}
        self.__cache['A0'] = X

        for l in range(1, self.__L + 1):
            Z = np.matmul(self.__weights['W' + str(l)],
                          self.__cache['A' + str(l - 1)]) + \
                          self.__weights['b' + str(l)]
            A = 1/(1+np.exp(-Z))
            self.__cache['A' + str(l)] = A

        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        """
        m = Y.shape[1]
        loss = - (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.sum(loss) / m

        return cost

    def evaluate(self, X, Y):
        """
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        """
        m = Y.shape[1]
        L = self.__L
        dZ = cache['A' + str(L)] - Y
        dW = np.matmul(dZ, cache['A' + str(L - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        W_prev = np.copy(self.__weights['W' + str(L)])
        self.__weights['W' + str(L)] -= alpha * dW
        self.__weights['b' + str(L)] -= alpha * db

        for l in range(L - 1, 0, -1):
            dA = np.matmul(W_prev.T, dZ)
            A = cache['A' + str(l)] 
            dZ = dA * A * (1 - A)
            dW = np.matmul(dZ, cache['A' + str(l - 1)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            W_prev = np.copy(self.__weights['W' + str(l)])
            self.__weights['W' + str(l)] -= alpha * dW
            self.__weights['b' + str(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if not 0 < step <= iterations:
                raise ValueError('step must be positive and <= iterations')

            if graph:
                x = np.arange(0, iterations + 1, step)
                size = iterations // step + 1
                if iterations % step:
                    size += 1
                    np.append(x, iterations)
                y = np.empty((size,))

        for i in range(0, iterations):
            A, cache = self.forward_prop(X)

            if (verbose or graph) and not i % step:
                cost = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    y[i // step] = cost

            self.gradient_descent(Y, cache, alpha)

        A, cost = self.evaluate(X, Y)

        if verbose:
            print("Cost after {} iterations: {}".format(iterations, cost))

        if graph:
            y[-1] = cost
            plt.plot(x, y)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return A, cost
