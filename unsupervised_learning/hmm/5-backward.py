#!/usr/bin/env python3
"""
comment for later
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    explain later
    """
    T = len(Observation)
    N, notit = Emission.shape
    B = np.zeros((N, T))
    B[:, -1] = 1
    for t in range(T - 2, -1, -1):
        for i in range(N):
            for j in range(N):
                B[i, t] = np.sum(B[j, t+1] * Transition[i, j] * Emission[j, Observation[t+1]])
    P = np.sum(B[i, 0] * Initial[i, 0] * Emission[i, Observation[0]] for i in range(N))
    return P, B
