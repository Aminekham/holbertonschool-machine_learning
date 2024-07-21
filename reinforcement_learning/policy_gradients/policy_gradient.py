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

def policy_gradient(state, weight):
    """
    Computing the needed gradient for
    the monte carlo policy gradient
    REINFORCE
    """
    probs = policy(state, weight)
    action = np.random.choice(len(probs[0]), p=probs[0])
    probs[0, action] -= 1
    grad = np.dot(state.T, probs)
    return action, grad
