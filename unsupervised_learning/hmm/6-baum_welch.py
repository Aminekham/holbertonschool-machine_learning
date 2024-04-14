#!/usr/bin/env python3
"""
for later
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    explanation later
    """
    T = len(Observations)
    M, N = Emission.shape

    for _ in range(iterations):
        alpha = np.zeros((M, T))
        alpha[:, 0] = np.squeeze(Initial * Emission[:, Observations[0]])
        for t in range(1, T):
            alpha[:, t] = np.dot(alpha[:, t-1], Transition) * Emission[:, Observations[t]]
        beta = np.zeros((M, T))
        beta[:, -1] = 1
        for t in range(T - 2, -1, -1):
            beta[:, t] = np.dot(Transition, Emission[:, Observations[t+1]] * beta[:, t+1])
        gamma = alpha * beta / np.sum(alpha * beta, axis=0)
        xi = np.zeros((M, M, T-1))
        for t in range(T - 1):
            xi[:, :, t] = (alpha[:, t].reshape(-1, 1) * Transition * Emission[:, Observations[t+1]].reshape(1, -1) * beta[:, t+1]) / np.sum(alpha[:, t] * beta[:, t])
        Transition = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1).reshape(-1, 1)
        gamma_sum = np.sum(gamma, axis=1).reshape(-1, 1)
        for k in range(N):
            mask = Observations == k
            Emission[:, k] = np.sum(gamma[:, mask], axis=1) / gamma_sum.ravel()
    return Transition, Emission
