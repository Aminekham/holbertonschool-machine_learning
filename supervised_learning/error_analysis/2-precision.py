#!/usr/bin/env python3
"""
calculating the precison of a certain
model (can be called positive predictive value)
"""

import numpy as np


def precision(confusion):
    """
    its calculated by a simple formula:
    precision = TP / (TP + FP)
    """
    classes = confusion.shape[0]
    precision = np.zeros((classes, ))
    sum_fp_list = np.sum(confusion, axis=0)
    for i in range(classes):
        tp = confusion[i][i]
        precision_i = tp / sum_fp_list[i]
        precision[i] = precision_i
    return(precision)
