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
