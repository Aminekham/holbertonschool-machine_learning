#!/usr/bin/env python3
"""
applying the learning rate
decay
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    decaying the learning rate
    so it gets to a minimum
    in the final steps of the gradient
    """
    new_alpha = alpha / (1 + decay_rate * np.floor(global_step / decay_step))
    return new_alpha
