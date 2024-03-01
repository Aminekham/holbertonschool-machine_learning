#!/usr/bin/env python3
"""
Convolutional Forward Prop
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    forward propagation:
    output height and width calculations show how the output
    dimensions changes using the simple equation:
    output dimensions = (input height - kernel height) + 1) *
    (input width - kernel width) + 1)
    """
    m, h_pr, w_pr, c_pr = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    output_h = int(((h_pr - kh) / sh) + 1)
    output_w = int(((w_pr - kw) / sw) + 1)
    if padding == "valid":
        A_prev = A_prev
    if padding == "same":
        pad_h = int(((h_pr - 1) * sh + kh - h_pr) // 2)
        pad_w = int(((w_pr - 1) * sw + kw - w_pr) // 2)
        A_prev = np.pad(A_prev, ((0, 0), (pad_h, pad_h),
                                 (pad_w, pad_w),
                                 (0, 0)),
                                 mode='constant',
                                 constant_values=0)
        output_h = h_pr
        output_w = w_pr

    A = np.zeros(shape=(m, output_h, output_w, c_new))
    for h in range(output_h):
        for w in range(output_w):
            for c in range(c_new):
                a_slice_prev = A_prev[:,h * sh : h * sh + kh,w * sw : w * sw + kw,:]
                W_c = W[:, :, :, c]
                A[:, h, w, c] = np.sum(a_slice_prev * W_c, axis=(1, 2, 3))                    
    return activation(A + b)
