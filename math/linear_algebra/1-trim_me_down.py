#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
for i in range(len(matrix)):
    the_middle.extend([matrix[i][j], matrix[i][j + 1]] for j in range(len(matrix[i])) if j == 2)
print("The middle columns of the matrix are: {}".format(the_middle))
