#!/usr/bin/env python3
"""
calculating the sensitivity(you can also call it recall or
true positive rate)
"""

import numpy as np


def sensitivity(confusion):
    """
    its calculated by a simple formula:
    sensitivity = TP / (TP + FN)
    """
    classes = confusion.shape[0]
    sensitivity = np.zeros((classes, ))
    for i in range(classes):
        fn = confusion[i].sum() - confusion[i][i]
        tp = confusion[i][i]
        sensitivity_i = tp / (tp + fn)
        sensitivity[i] = sensitivity_i
    return(sensitivity)
