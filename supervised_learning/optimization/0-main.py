#!/usr/bin/env python3

import numpy as np

update_variables_RMSProp = __import__('7-RMSProp').update_variables_RMSProp


np.random.seed(8)
a, b = np.random.uniform(low=0.01, size=2)
m, nv = np.random.randint(10, 100, 2)
v = np.random.randn(m, nv)
dv = np.random.randn(m, nv)
s = np.random.uniform(m, nv)
print(update_variables_RMSProp(a, b, 1e-8, v, dv, s))