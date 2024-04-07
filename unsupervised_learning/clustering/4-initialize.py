#!/usr/bin/env python
"""

"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans



def initialize(X, k):
    pi = np.full((k,), 1 / k)
    try:
        centroids, _ = kmeans(X, k)
    except Exception:
        return None, None, None
    d = X.shape[1]
    covariances = np.tile(np.eye(d), (k, 1, 1))
    return pi, centroids, covariances
