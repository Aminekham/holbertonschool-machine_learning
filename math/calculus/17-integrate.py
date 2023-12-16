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
    integral = [C]

    for i in range(len(poly)):
        term = poly[i] / (i + 1)
        if term.is_integer():
            term = int(term)
        integral.append(term)

    return integral
