#!/usr/bin/env python3
""" This is a function to add arrays """
shape = __import__('2-size_me_please').matrix_shape


def add_arrays(arr1, arr2):
    """
        1- testing the obligatory shape condition
        adding_result: varibale to store the final addition array
    """
    if shape(arr1) != shape(arr2):
        return(None)
    adding_result = []
    for i in range(len(arr1)):
        adding_result.append(arr1[i] + arr2[i])
    return(adding_result)
