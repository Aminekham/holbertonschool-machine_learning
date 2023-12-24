#!/usr/bin/env python3
"""
This is our main exponential class
"""


class Normal:
    """
    this defines the exponential class to
    understand the core concepts of
    exponential distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        init function to check values and calculate a reasonable value for
        the lambtha rate
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            n = len(data)
            self.mean = float(sum(data) / len(data))
            self.stddev = (sum((x - self.mean) ** 2 for x in data) / n) ** 0.5

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
