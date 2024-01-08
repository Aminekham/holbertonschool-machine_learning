#!/usr/bin/env python3

import numpy as np
Neuron = __import__('1-neuron').Neuron

np.random.seed(2)
nx = np.random.randint(100, 1000)
nn = Neuron(nx)
print(nn.W)
print(nn.b)
print(nn.A)