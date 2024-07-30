#!/usr/bin/env python3
"""

"""
import pandas as pd


def from_numpy(array):
    """

    """
    num_columns = array.shape[1]
    columns = [chr(i) for i in range(65, 65 + num_columns)]
    df = pd.DataFrame(array, columns=columns)    
    return df
