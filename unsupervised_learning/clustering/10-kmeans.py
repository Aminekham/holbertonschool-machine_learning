#!/usr/bin/env python3
"""
K-means implementation
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Using sklearn to perform the
    already implemented kmeans
    algorithm
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
