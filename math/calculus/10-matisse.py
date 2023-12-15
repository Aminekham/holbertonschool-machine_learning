#!/usr/bin/env python3
"""
function to get the sum of i squared
from 1 to the given n
"""


def poly_derivative(poly):
    derivarive = []
    for i in range(len(poly)):
        derivarive.append(poly[i] * i)
    if sum(derivarive) == 0:
        return([0])
    return(derivarive[1:])
