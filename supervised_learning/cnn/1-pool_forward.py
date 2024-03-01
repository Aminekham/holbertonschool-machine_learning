#!/usr/bin/env python3
"""
Perform a forward pass of the pooling layer on A_prev
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Perform a forward pass of the pooling layer on A_prev
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h_prev - kh) // sh + 1
    output_w = (w_prev - kw) // sw + 1

    A_output = np.zeros(shape=(m, output_h, output_w, c_prev))
    for h in range(output_h):
        for w in range(output_w):
            for _ in range(c_prev):
                if mode == 'max':
                    A_output[:, h, w, :] = np.max(A_prev[:,
                                                         h * sh: h * sh + kh,
                                                         w * sw: w * sw + kw,
                                                         :],
                                                  axis=(1, 2))
                else:
                    A_output[:, h, w, :] = np.mean(A_prev[:,
                                                          h * sh: h * sh + kh,
                                                          w*sw: w*sw+kw, :],
                                                   axis=(1, 2))
    return A_output
