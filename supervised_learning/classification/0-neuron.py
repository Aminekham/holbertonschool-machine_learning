#!/usr/bin/env python3
import numpy as np
"""
Creating the neural network by our
own
"""


class Neuron:
    """
    this is the neuron responsible for performing 
    the classification task 
    """
    def __init__(self, nx):
        """
        initialisation function to get the 
        weights entering the neuron
        while defining the bias b and the output A
        """
        self.W = np.random.normal(nx)
        self.b = 0
        self.A = 0
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
