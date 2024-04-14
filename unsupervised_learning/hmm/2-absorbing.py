#!/usr/bin/env python3
"""
Cheking if a markov chain
have an absorbing state or not
"""
import numpy as np


def absorbing(P):
    """
    an absorbing state is a state
    that have a 1 probability to itself
    which means like its a black hole
    """
    length = len(P)
    for i in range(length):
        if P[i, i] == 1:
            return True
    return False
