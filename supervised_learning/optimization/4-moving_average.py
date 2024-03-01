#!/usr/bin/env python3
"""
determening the moving slide
"""


def moving_average(data, beta):
    """
    the moving average is when
    we have all the possible pieces of
    a data and return it while averaging
    it by a certain rate
    """
    m_avg = []
    weighted_sum = 0.0
    correction_sum = 0.0
    for i in range(len(data)):
        weighted_sum = beta * weighted_sum + (1 - beta) * data[i]
        correction_sum = 1 - (beta ** (i + 1))
        corrected_moving = weighted_sum / correction_sum
        m_avg.append(corrected_moving)

    return m_avg
