#!/usr/bin/env python3

"""
This is our main poisson class
"""
class Poisson:
    """
    creating the class to work on poisson distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        init function to check values and calculate a reasonable value for 
        the poisson rate
        """
        self.lambtha = float(lambtha)
        if data == []:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        self.lambtha = sum(data) / len(data)
        if type(data) != list:
            raise TypeError("data must be a list")
        if len(data) < 2:
            raise ValueError("data must contain multiple values")
