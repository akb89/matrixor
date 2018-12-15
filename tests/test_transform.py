import numpy as np

import matrixor.transformation.transformator as mtrans


def test_center():
    matrix = np.array([[1, 1], [3, 1], [1, 3], [3, 3]])
    centered = mtrans.center(matrix)
    np.testing.assert_array_equal(
        centered, [[-1, -1], [1, -1], [-1, 1], [1, 1]])


def test_sum_outer_product():
    A = np.array([[0, 1, 0], [1, 0, 1]])
    B = np.array([[0, 0, 0], [1, 1, 1]])
    C = mtrans.sum_outer_product(A, B)
    np.testing.assert_array_equal(
        C, [[1, 0, 1], [1, 0, 1], [1, 0, 1]])
