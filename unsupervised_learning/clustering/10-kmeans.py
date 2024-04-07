#!/usr/bin/env python3
"""
K-means implementation
"""
from sklearn.cluster import KMeans


def kmeans(X, k):
    """
    Using sklearn to perform the
    already implemented kmeans
    algorithm
    """
    kmeans = KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
