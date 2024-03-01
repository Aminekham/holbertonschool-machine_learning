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
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        pmf function to calculate the probility mass function
        of a certain success k
        exp: telling how much in the intervale of lambtha can the k
        event occur
        """
        k = int(k)
        if k < 0:
            return 0
        return((2.7182818285 ** -self.lambtha * self.lambtha ** k)
               / factorial(k))

    def cdf(self, k):
        """
        cdf function to give the pourcentile of
        an event to score less than that
        """
        k = int(k)
        if k < 0:
            return 0
        return self.pmf(k) + self.cdf(k - 1)


def factorial(n):
    """
        needed to calculate the factorial for the pmf
        equation
    """
    if n == 0:
        return(1)
    return(n * factorial(n - 1))
