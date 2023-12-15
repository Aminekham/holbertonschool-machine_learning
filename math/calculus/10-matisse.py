#!/usr/bin/env python3
"""
function to get the derivative of a polynominal function
"""


def poly_derivative(poly):
    """
    Parameters: 
        derivative: list to save the coefficents of the derivative
    """
    derivarive = []
    if type(poly) != list:
        return(None)
    for i in range(len(poly)):
        derivarive.append(poly[i] * i)
    if sum(derivarive) == 0:
        return([0])
    return(derivarive[1:])
