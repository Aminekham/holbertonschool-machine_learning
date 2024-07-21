#!/usr/bin/env python3
"""
Applying the simple
change from weights
and state matrix to
probabilities for the agent
using softmax function ofc
"""
import numpy as np

def policy(matrix, weight):
    """
    - computing the scores to have the
    unormalized probability of each action given a certain state
    - normalize
    - applying the softmax function to get probabilities
    """
    z = np.exp(np.dot(matrix, weight) - np.max(np.dot(matrix, weight)))
    policy = z / np.sum(z)
    return policy
