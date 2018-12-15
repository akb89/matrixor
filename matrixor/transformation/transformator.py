"""Matrix transformations."""

import numpy as np

__all__ = ('center', 'sum_outer_product')


def center(matrix):
    """Center input matrix."""
    return matrix - matrix.mean(axis=0)

def sum_outer_product(A, B):
    return np.einsum('ij,ik->jk', B, A)

def align_ao_centered(A, B):
    AC = center(A)
    BC = center(B)
    H = sum_outer_product(AC, BC)
    U, S, VT = np.linalg.svd(H)
    R = H*VT  # rotation matrix
    RBC = BC * R  # rotated centered B
    return AC, RBC
