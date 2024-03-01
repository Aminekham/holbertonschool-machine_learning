#!/usr/bin/env python3
"""
applying the convolution
with a grayscale kernal
on input images
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    taking the height and width of each image and
    the kernel and then applying the convolution
    each time on the selected image region and so
    on with addition of padding as a way to avoid the
    negative values and replace them with 0
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    result = np.zeros((m, output_h, output_w))
    pad_h = kh - 1
    pad_w = kw - 1
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded_images = np.pad(images, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    for i in range(output_h):
        for j in range(output_w):
            padded_image = padded_images[:, i:i+kh, j:j+kw]
            result[:, i, j] = np.sum(padded_image * kernel, axis=(1, 2))
    return result
