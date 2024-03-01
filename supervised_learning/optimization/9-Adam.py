#!/usr/bin/env python3
"""
Using adam optimizer
which is a combination
"""


import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Update variables using Adam optimization algorithm
    """
    m_t = beta1 * v + (1 - beta1) * grad
    v_t = beta2 * s + (1 - beta2) * grad ** 2
    m_t = m_t / (1 - beta1 ** t)
    v_t = v_t / (1 - beta2 ** t)
    var = var - (alpha / (np.sqrt(v_t)) + epsilon) * m_t
    return var, m_t, v_t
