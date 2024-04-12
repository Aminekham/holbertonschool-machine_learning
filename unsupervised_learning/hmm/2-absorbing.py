#!/usr/bin/env python3
"""

"""
import numpy as np


def absorbing(P):
    """

    """
    length = len(P)
    for i in range(length):
        if P[i, i] == 1:
            return True
    return False
