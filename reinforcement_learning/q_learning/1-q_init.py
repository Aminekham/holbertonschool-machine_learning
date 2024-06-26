#!/usr/bin/env python3
import numpy as np
"""
Init for the Q-table
"""
def q_init(env):
    """
    Q-table as a numpy.ndarray of zeros with shape
    (number_of_states, number_of_actions)
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
