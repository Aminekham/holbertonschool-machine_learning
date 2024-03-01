#!/usr/bin/env python3
"""
calculating the specificity of
a certain classification model
its like precision but for negative
values
"""

import numpy as np


def specificity(confusion):
    """
    its calculated by a simple formula:
    specificity = TN / (TN + FN)
    """
    classes = confusion.shape[0]
    specificity = np.zeros((classes,))
    for i in range(classes):
        true_negatives = np.sum(confusion) - np.sum(confusion[i, :])
        true_negatives = true_negatives - np.sum(confusion[:, i])
        true_negatives = true_negatives + confusion[i, i]
        false_positives = np.sum(confusion[:, i]) - confusion[i, i]
        specificity_i = true_negatives / (true_negatives + false_positives)
        specificity[i] = specificity_i
    return specificity
