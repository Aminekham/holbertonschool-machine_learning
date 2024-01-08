#!/usr/bin/env python3

import numpy as np
Neuron = __import__('3-neuron').Neuron

np.random.seed(4)
nx, m = np.random.randint(100, 1000, 2).tolist()
nn = Neuron(nx)
Y = np.random.randint(0, 2, (1, m))
A = np.random.uniform(size=(1, m))
print(np.round(nn.cost(Y, A), 10))