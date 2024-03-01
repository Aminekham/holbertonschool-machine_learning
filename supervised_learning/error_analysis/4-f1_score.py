#!/usr/bin/env python3
"""
calculating the f1 score which
is the combination of sensitivity
and precision
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    its calculated by a simple formula:
    f1_score = 2 *(precision * sensitivity) / (sensitivity + precision)
    """
    precision_matrix = precision(confusion)
    sensitivity_matrix = sensitivity(confusion)
    classes = confusion.shape[0]
    f1_score = np.zeros((classes, ))
    for i in range(classes):
        if sensitivity_matrix[i] + precision_matrix[i] == 0:
            f1_score[i] = 0
        f1_score[i] = 2 * (precision_matrix[i] * sensitivity_matrix[i])
        f1_score[i] = f1_score[i] / (precision_matrix[i] + sensitivity_matrix[i])
    return f1_score
