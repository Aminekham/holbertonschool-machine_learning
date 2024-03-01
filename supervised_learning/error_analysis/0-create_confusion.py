#!/usr/bin/env python3
"""
the confusion matrix is a table that describes the performance of an
algorithm on a classification problem
it provides insight into how the predictions
are classified into TP, TN, FP, FN
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    using the approach of calculating
    what did the model give as results
    and this is widely used for multinominal
    logistic regression for example
    """
    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))
    for i in range(labels.shape[0]):
        true_class = np.argmax(labels[i])
        predicted_class = np.argmax(logits[i])
        confusion[true_class, predicted_class] += 1
    return confusion
