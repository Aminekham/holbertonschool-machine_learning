#!/usr/bin/env python3
"""
Gaussian process
"""
import numpy as np


class GaussianProcess:
    """
    Gaussian process class
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        init function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)
    def kernel(self, X1, X2):
            """
            calculating the covariance kernel
            which is basically the eucleudian distance
            """
            k = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(X1.shape[0]):
                for j in range(X2.shape[0]):
                    distance = np.linalg.norm(X1[i] - X2[j])
                    k[i, j] = np.exp(-distance ** 2 / (2 * self.l ** 2))
            return k
