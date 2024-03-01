#!/usr/bin/env python3

import numpy as np
Neuron = __import__('6-neuron').Neuron

np.random.seed(7)
nx, m = np.random.randint(100, 1000, 2).tolist()
nn = Neuron(nx)
X = np.random.randn(nx, m)
Y = np.random.randint(0, 2, (1, m))
A, cost = nn.train(X, Y)
print(np.round(A, decimals=10))
print(np.round(cost, decimals=10))
print(np.round(nn.W, decimals=10))
print(np.round(nn.b, decimals=10))
print(np.round(nn.A, decimals=10))
try:
    nn.A = 10
    print('Fail: Private attribute A overwritten as public attribute')
except:
    pass
try:
    nn.W = 10
    print('Fail: Private attribute W overwritten as public attribute')
except:
    pass
try:
    nn.b = 10
    print('Fail: Private attribute b overwritten as public attribute')
except:
    pass