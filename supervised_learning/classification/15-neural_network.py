#!/usr/bin/env python3
"""
"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    """

    def __init__(self, nx, nodes):
        """
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes <= 0:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0

        self.__A1 = 0
        self.__A2 = 0

    @property
    def A1(self):
        return self.__A1

    @property
    def A2(self):
        return self.__A2

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    def forward_prop(self, X):
        """
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1/(1+np.exp(-Z1))
        Z2 = np.matmul(self.W2, self.__A1) + self.__b2
        self.__A2 = 1/(1+np.exp(-Z2))

        return self.__A1, self.__A2

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
        _, A = self.forward_prop(X)
        cost = self.cost(Y, A)

        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        """
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.matmul(self.__W2.T, dZ2)*A1*(1-A1)
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2

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
            A1, A2 = self.forward_prop(X)

            if (verbose or graph) and not i % step:
                cost = self.cost(Y, A2)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    y[i // step] = cost

            self.gradient_descent(X, Y, A1, A2, alpha)

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
