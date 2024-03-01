#!/usr/bin/env python3
"""
applying the convolution
with a grayscale kernal
on input images
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    taking the height and width of each image and
    the kernel and then applying the convolution
    each time on the selected image region and so
    on
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    result = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            image_region = images[:, i:i+kh, j:j+kw]
            result[:, i, j] = np.sum(image_region * kernel, axis=(1, 2))
    return result
