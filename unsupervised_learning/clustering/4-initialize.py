#!/usr/bin/env python3
"""

"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans



def initialize(X, k):
    pi = np.full((k,), 1 / k)
    try:
        centroids, _ = kmeans(X, k)
        d = X.shape[1]
        covariances = np.tile(np.eye(d), (k, 1, 1))
        return pi, centroids, covariances
    except Exception:
        return None, None, None
