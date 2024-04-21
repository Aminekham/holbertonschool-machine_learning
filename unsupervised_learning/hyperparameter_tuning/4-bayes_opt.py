#!/usr/bin/env python3
"""
The bayessian optimization
class file
"""
from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    the bayessian optimization class which is
    using the gaussian  process method to pick the best possible parameters
    based on an accusation function and optimizing it
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Initializing the bayessian optimization parameters + GP object
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        defining the acquisition function u
        based on the expected improvement (EI)
        u = (mu(x) - u+ - epsilon) cdf(Z) + sigma(Z) pdf(Z)
        """
        mu, var = self.gp.predict(self.X_s)
        if self.minimize is True:
            Y_sample = np.min(self.gp.Y)
            imp = Y_sample - mu - self.xsi
        else:
            Y_sample = np.max(self.gp.Y)
            imp = mu - Y_sample - self.xsi
        Z = np.zeros(var.shape[0])
        for i in range(var.shape[0]):
            if var[i] > 0:
                Z[i] = imp[i] / var[i]
            else:
                Z[i] = 0
            ei = imp * norm.cdf(Z) + var * norm.pdf(Z)
        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei
