#!/usr/bin/env python3
"""
function to get the sum of i squared
from 1 to the given n
"""


def summation_i_squared(n):
    """
    Parameters:
        sum: The variable storing the square of i
        each time
    """
    if not isinstance(n, int) or n < 1:
        return None
    if n == 1:
        return 1
    return n**2 + summation_i_squared(n-1)
