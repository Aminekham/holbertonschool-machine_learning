#!/usr/bin/env python3
"""
This is our main exponential class
"""


class Exponential:
    """
    this defines the exponential class to
    understand the core concepts of
    exponential distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        init function to check values and calculate a reasonable value for
        the lambtha rate
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / float(sum(data) / len(data))

    def pdf(self, x):
        """
        probability distribution function to search for the
        likelihood of a particular
        random variable to be in a the needed range of values
        """
        if x < 0:
            return 0
        return(self.lambtha * 2.7182818285 ** - (self.lambtha * x))

    def cdf(self, x):
        """
        cumulative distribution function to get the probability
        of our random variable is less or equal to the value of x
        """
        if x < 0:
            return 0
        return(1 - 2.7182818285 ** - (self.lambtha * x))
