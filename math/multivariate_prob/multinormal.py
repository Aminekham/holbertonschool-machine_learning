#!/usr/bin/env python3
"""
multinormal
"""
import numpy as np


class MultiNormal:
    """
    class description
    """
    def __init__(self, data):
        """
        init function
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        centered_data = data - self.mean
        self.cov = np.dot(centered_data, centered_data.T) / (data.shape[1] - 1)
    def pdf(self, x):
        """
        calculating the pdf value
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.mean.shape[0], 1):
            raise ValueError(f"x must have the shape ({self.mean.shape[0]}, 1)")
        exponent = -0.5 * np.dot((x - self.mean).T, np.dot(self.inv_cov, (x - self.mean)))
        pdf_value = (1 / np.sqrt((2 * np.pi) ** self.mean.shape[0] * self.det_cov)) * np.exp(exponent)
        return pdf_value

