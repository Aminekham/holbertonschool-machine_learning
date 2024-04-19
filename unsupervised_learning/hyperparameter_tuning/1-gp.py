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
        sqdist1 = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1)
        sqdist2 = 2 * np.dot(X1, X2.T)
        sqdist = sqdist1 - sqdist2
        k = self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
        return k

    def predict(self, X_s):
        """
        Calculating the mean and
        std div
        equation for mean
        mean = K_training_test_data.T*(K_training+squared(sigma)*I)^-1*y_training
        """
        cov_training = self.kernel(self.X, self.X)
        cov_training_test = self.kernel(self.X, X_s)
        cov_test = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)
        mu = cov_training_test.T.dot(K_inv).dot(self.Y)
        mu = np.reshape(mu, -1)
        sigma = cov_test - cov_training_test.T.dot(K_inv).dot(cov_training_test)
        sigma = sigma.diagonal()
        return mu, sigma
