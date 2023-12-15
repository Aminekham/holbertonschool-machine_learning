#!/usr/bin/env python3
"""
function to get the sum of i squared
from 1 to the given n
"""
import numpy as np


def summation_i_squared(n):
    """
    Parameters: 
        sum: The variable storing the square of i
        each time
    """
    if not isinstance(n, int) or n < 1:
        return None
    numbers = np.arange(1, n+1)
    return np.sum(numbers**2)
