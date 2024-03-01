#!/usr/bin/env python3
"""
Applying stride(step of the kernel)
took for every movement of that kernel
calculated by pixels
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    taking the height and width of each image and
    the kernel and then applying the convolution
    each time on the selected image region and so
    on with addition of padding as a way to avoid the
    negative values and replace them with 0 and
    applying a stride to it as its the step of which
    the kernel moves by pixels
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'valid':
        pad_h, pad_w = 0, 0
    elif padding == 'same':
        pad_h = int(((h - 1) * sh + kh - h) / 2) + 1
        pad_w = int(((w - 1) * sw + kw - w) / 2) + 1
    elif isinstance(padding, tuple) and len(padding) == 2:
        pad_h, pad_w = padding
    else:
        raise ValueError("Invalid padding value")
    padded_images = np.pad(images, ((0, 0),
                                    (pad_h, pad_h),
                                    (pad_w, pad_w)),
                                    mode='constant',
                                    constant_values=0)
    output_h = int((h + 2 * pad_h - kh) / sh) + 1
    output_w = int((w + 2 * pad_w - kw) / sw) + 1
    result = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            result[:, i, j] = np.sum(padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw] * kernel, axis=(1, 2))
    return result
