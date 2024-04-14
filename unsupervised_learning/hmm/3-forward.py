#!/usr/bin/env python3
"""

"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    
    """
    T = len(Observation)
    N, M = Emission.shape
    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[i, t-1] * Transition[i, j] * Emission[j, Observation[t]] for i in range(N))
    P = np.sum(F[:, -1])
    return P, F
