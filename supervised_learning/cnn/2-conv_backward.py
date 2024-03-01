#!/usr/bin/env python3
"""
Compute backpropagation for a convolutional layer
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    this computation is based how the backpropagation algorithm is based
    lets dive into its mathematics:
    the needed values are :
    dA_prev: which is the derivative of the loss function
    with respect to the previous layer
    to be completed
    """
    m, h_new, w_new, c_new = dZ.shape
    m1, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    if padding == "valid":
        A_prev_padded = A_prev
    else:
        pad_h = int(((h_prev - 1) * sh + kh - h_prev) // 2)
        pad_w = int(((w_prev - 1) * sw + kw - w_prev) // 2)
        A_prev_padded = np.pad(A_prev,
                               ((0, 0),
                                (pad_h, pad_h),
                                (pad_w, pad_w), (0, 0)),
                                mode='constant', constant_values=0)
    for example in range(m):
        for channel in range(c_new):
            for h in range(h_new):
                for w in range(w_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    a_slice = A_prev_padded[example,
                                            vert_start:vert_end,
                                            horiz_start:horiz_end,
                                            :]
                    dA_prev[example, 
                            vert_start:vert_end,
                            horiz_start:horiz_end,:] += W[:,
                                                          :, :, channel] * dZ[example,
                                                           h, w, channel]
                    dW[:, :, :, channel] += a_slice * dZ[example,
                                                         h, w, channel]
                    db[:, :, :, channel] += dZ[example, h, w, channel]
    return dA_prev, dW, db
