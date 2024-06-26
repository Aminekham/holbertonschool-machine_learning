#!/usr/bin/env 
"""
Using epsilon greedy
technique
"""
import numpy as np
def epsilon_greedy(Q, state, epsilon):
    """
    Exploration probability is epsilon
    Expoitation probability is 1-epsilon
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.argmax(Q[state, :])
    if p > epsilon:
        action = action = np.random.randint(0, Q.shape[1])
    return action
