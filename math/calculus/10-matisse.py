#!/usr/bin/env python3
"""
function to get the derivative of a polynominal function
"""


def poly_derivative(poly):
    """
    Parameters:
        derivative: list to save the coefficents of the derivative
    """
    if not isinstance(poly, list):
        return None
    if not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None
    derivative = []
    for i in range(len(poly)):
        derivative.append(poly[i] * i)
    if sum(derivative) == 0:
        return [0]
    return derivative[1:]
