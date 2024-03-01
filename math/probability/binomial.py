#!/usr/bin/env python3
"""
This is our main binomial class
"""


class Binomial:
    """
    this defines the binomial class to
    understand the core concepts of
    binomial distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        init function to check values and calculate a reasonable value for
        the number of trials and probability of success
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(round(n))
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            p = sum(data) / (len(data) * self.n)
            self.n = int(round(len(data) / (1 - p)))
            self.p = sum(data) / (len(data) * self.n)
