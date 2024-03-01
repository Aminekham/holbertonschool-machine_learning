#!/usr/bin/env python3
"""
creating the early stopping process
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    defining the early stopping and
    how does it work
    """
    if cost > opt_cost - threshold:
        count += 1
    else:
        count = 0
    return (count >= patience, count)
