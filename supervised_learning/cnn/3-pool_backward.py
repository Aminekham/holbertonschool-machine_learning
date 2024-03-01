#!/usr/bin/env python3
"""
The backpropagation algorithm to the
pooling layer
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    pooling is an operation which means
    that it applies directly and derivating it means
    reversing the process (getting the maximum and how much
    did it affect the output also getting the average and how
    much did the input values affect the averages
    by how much it was used in the pooling proccess)
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for example in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for channel in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    if mode == 'max':
                        a_prev_slice = A_prev[example,
                                              vert_start:vert_end,
                                              horiz_start:horiz_end,
                                              channel]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[example, vert_start:vert_end,
                                horiz_start:horiz_end,
                                channel] += mask * dA[example,
                                                      h, w, channel]
                    else:
                        da = dA[example, h, w, channel]
                        average = da / (kh * kw)
                        dA_prev[example,
                                vert_start:vert_end,
                                horiz_start:horiz_end,
                                channel] += np.ones((kh, kw)) * average

    return dA_prev
