#!/usr/bin/env python3
"""
function to get the derivative of a polynominal function
"""


def poly_integral(poly, C=0):
    """
    Parameters:
        derivative: list to save the coefficents of the derivative
    """
    if not isinstance(poly, list):
        return None
    if not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None
    integral = []
    integral.append(0)
    for i in range(len(poly)):
        integral.append(poly[i] / (i+1))
    if sum(integral) == 0:
        return [0]
    return integral
