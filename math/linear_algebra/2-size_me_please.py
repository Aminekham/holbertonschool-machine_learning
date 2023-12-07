#!/usr/bin/env python3
def matrix_shape(matrix):
    """
    Get the shape of a matrix with any dimensionality.

    Parameters:
    - matrix (list): The input matrix.

    Returns:
    - list: A list representing the shape of the matrix. Each element in the list
            corresponds to the size of a dimension.
    """

    # Make a temporary copy of the matrix to avoid modifying the original
    temp = matrix

    # Initialize an empty list to store the shape of the matrix
    shape = []

    # The first element of the shape is the number of rows in the matrix
    shape.append(len(temp))

    # Iterate through each level of nested lists to get the size of each dimension
    for i in range(len(temp)):
        # Check if the elements in the current level are lists (nested dimension)
        if type(temp[0]) != list:
            break

        # Append the size of the current dimension to the shape list
        shape.append(len(temp[i]))

        # Update the temporary variable to the current level for the next iteration
        temp = temp[i]

    # Return the final shape of the matrix
    return shape
