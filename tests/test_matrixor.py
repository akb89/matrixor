import math

import numpy as np
import scipy.stats as stats

import matrixor


def pearson_correlation(x, y):
    return stats.pearsonr(x, y)[0]


def cosine_similarity(peer_v, query_v):
    if len(peer_v) != len(query_v):
        raise ValueError('Vectors must be of same length')
    num = np.dot(peer_v, query_v)
    den_a = np.dot(peer_v, peer_v)
    den_b = np.dot(query_v, query_v)
    return num / (math.sqrt(den_a) * math.sqrt(den_b))


def test_center():
    matrix = np.array([[1, 1], [3, 1], [1, 3], [3, 3]])
    centered = matrixor.center(matrix)
    np.testing.assert_array_equal(
        centered, [[-1, -1], [1, -1], [-1, 1], [1, 1]])


def test_sum_outer_product():
    A = np.array([[0, 1, 0]])
    B = np.array([[1, 1, 0]])
    C = matrixor.compute_sum_outer_product(A, B)
    np.testing.assert_array_equal(
        C, [[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    A = np.array([[0, 1, 0], [1, 0, 1]])
    B = np.array([[0, 0, 0], [1, 1, 1]])
    C = matrixor.compute_sum_outer_product(A, B)
    np.testing.assert_array_equal(
        C, [[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    A = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 1]])
    B = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    C = matrixor.compute_sum_outer_product(A, B)
    np.testing.assert_array_equal(
        C, [[1, 2, 1], [1, 2, 1], [1, 2, 1]])


def test_sum_inner_product():
    A = np.array([[0, 1, 0]])
    B = np.array([[1, 1, 0]])
    c = matrixor.compute_sum_inner_product(A, B)
    assert c == 1
    A = np.array([[0, 1, 0], [1, 0, 1]])
    B = np.array([[0, 0, 0], [1, 1, 1]])
    c = matrixor.compute_sum_inner_product(A, B)
    assert c == 2
    A = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 1]])
    B = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    c = matrixor.compute_sum_inner_product(A, B)
    assert c == 4


def test_ao_rotation():
    A = np.array([[0, 1, 0], [1, 1, 1], [1, 1, 0]])
    B = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    BR = matrixor.apply_ao_rotation(A, B)
    assert abs(cosine_similarity(B[0], B[1]) - cosine_similarity(BR[0], BR[1])) < 0.000001
    assert abs(cosine_similarity(B[1], B[2]) - cosine_similarity(BR[1], BR[2])) < 0.000001
    AR = matrixor.apply_ao_rotation(B, A)
    assert abs(cosine_similarity(A[0], A[1]) - cosine_similarity(AR[0], AR[1])) < 0.000001
    assert abs(cosine_similarity(A[1], A[2]) - cosine_similarity(AR[1], AR[2])) < 0.000001


def test_apply_ao_with_scaling():
    A = np.array([[0, 1, 0], [1, 1, 1], [1, 1, 0]])
    B = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    T = matrixor.apply_absolute_orientation_with_scaling(A, B)
    assert abs(cosine_similarity(B[0], B[1]) - cosine_similarity(T[0], T[1])) < 0.000001
    assert abs(cosine_similarity(B[1], B[2]) - cosine_similarity(T[1], T[2])) < 0.000001
    U = matrixor.apply_absolute_orientation_with_scaling(B, A)
    assert abs(cosine_similarity(A[0], A[1]) - cosine_similarity(U[0], U[1])) < 0.000001
    assert abs(cosine_similarity(A[1], A[2]) - cosine_similarity(U[1], U[2])) < 0.000001
    assert abs(matrixor.root_mean_square_error(A, T) - matrixor.root_mean_square_error(U, B)) < 0.000001

def test_ao_scaling_in_diff_config():
    A = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    B = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
    T = matrixor.apply_absolute_orientation_with_scaling(A, B)
    assert matrixor.root_mean_square_error(A, T) < 1e-10
    A = np.array([[1, 0], [1, 0], [1, 0]])
    B = np.array([[0, 1], [0, 1], [0, 1]])
    T = matrixor.apply_absolute_orientation_with_scaling(A, B)
    assert matrixor.root_mean_square_error(A, T) < 1e-10
    A = np.array([[1, 0], [1, 0], [0, 1]])
    B = np.array([[0, 1], [0, 1], [1, 0]])
    T = matrixor.apply_absolute_orientation_with_scaling(A, B)
    assert matrixor.root_mean_square_error(A, T) < 1e-10
    A = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    B = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    T = matrixor.apply_absolute_orientation_with_scaling(A, B)
    assert matrixor.root_mean_square_error(A, T) < 1e-10
    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    T = matrixor.apply_absolute_orientation_with_scaling(A, B)
    assert matrixor.root_mean_square_error(A, T) < 1e-10

    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
    #print(pearson_correlation(A[:, 0], B[:, 0]))
    T = matrixor.apply_absolute_orientation_with_scaling(A, B)
    assert matrixor.root_mean_square_error(A, T) < 1e-15

    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([[.9, 0, 0, 0], [0, .9, 0, 0], [0, 0, .9, 0], [0, 0, 0, .9]])
    print(pearson_correlation(A[:, 0], B[:, 0]))
    T = matrixor.apply_absolute_orientation_with_scaling(A, B)
    assert matrixor.root_mean_square_error(A, T) < 1e-15

    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([[0, 0, 0, -1], [0, 0, -1, 0], [-1, 0, 0, 0], [0, -1, 0, 0]])
    T = matrixor.apply_absolute_orientation_with_scaling(A, B)
    assert matrixor.root_mean_square_error(A, T) < 1e-10
    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0]])
    T = matrixor.apply_absolute_orientation_with_scaling(A, B)
    assert matrixor.root_mean_square_error(A, T) < 1e-10
    A = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    B = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0]])
    T = matrixor.apply_absolute_orientation_with_scaling(A, B)
    assert matrixor.root_mean_square_error(A, T) < 1e-10
    A = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([[0, 0, 0, -1], [0, 0, -1, 0], [-1, 0, 0, 0], [0, -1, 0, 0]])
    C = np.abs(A - B)
