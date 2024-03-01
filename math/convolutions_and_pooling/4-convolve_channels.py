#!/usr/bin/env python3
"""

"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernel.shape
    sh, sw = stride
    if padding == 'valid':
        pad_h = 0
        pad_w = 0
    elif padding == 'same':
        pad_h = int(((h - 1) * sh + kh - h) / 2) + 1
        pad_w = int(((w - 1) * sw + kw - w) / 2) + 1
    elif isinstance(padding, tuple) and len(padding) == 2:
        pad_h, pad_w = padding
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
    output_h = int((h + 2 * pad_h - kh) / sh) + 1
    output_w = int((w + 2 * pad_w - kw) / sw) + 1
    result = np.zeros((m, output_h, output_w, nc))
    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                result[:, i, j, k] = np.sum(padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :] * kernel[:, :, :, k])
    return result
