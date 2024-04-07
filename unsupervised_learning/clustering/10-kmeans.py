#!/usr/bin/env python3
"""

"""
import numpy as np
from sklearn.cluster import KMeans


def kmeans(X, k):
    """
    
    """
    kmeans = KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
