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
    sum = 1
    for i in range(2, n+1):
        sum += i ** 2
    return(sum)
