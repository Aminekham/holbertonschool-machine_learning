#!/usr/bin/env python3
"""
This is our main normal class
"""


class Normal:
    """
    this defines the normal class to
    understand the core concepts of
    normal distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        init function to check values and calculate a reasonable value for
        the mean and standard diviation
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

    def z_score(self, x):
        z = (x - self.mean) / self.stddev
        return(z)
    def x_value(self, z):
        x = z * self.stddev + self.mean
        return(x)