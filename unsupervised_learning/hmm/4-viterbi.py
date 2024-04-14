#!/usr/bin/env python3
"""
comment for later
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    explain here later
    """
    T = len(Observation)
    N, notit = Emission.shape
    V = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        for j in range(N):
            x = V[:, t-1] * Transition[:, j]
            y = Emission[j, Observation[t]]
            probabilities = x * y
            V[j, t] = np.max(probabilities)
            backpointer[j, t] = np.argmax(probabilities)
    P = np.max(V[:, -1])
    last_state = np.argmax(V[:, -1])
    path = [last_state]
    for t in range(T - 1, 0, -1):
        last_state = backpointer[last_state, t]
        path.insert(0, last_state)
    return path, P
