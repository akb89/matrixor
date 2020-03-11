"""Welcome to matrixor.

This is the entry point of the application.
"""

import os

import argparse
import logging
import logging.config

import numpy as np
from einsumt import einsumt

import matrixor.utils.config as cutils


logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)

__all__ = ('apply_absolute_orientation_with_scaling')


def center(matrix):
    """Center input matrix."""
    return matrix - matrix.mean(axis=0)


def compute_sum_inner_product(A, B):
    # return np.diagonal(A @ B.T).sum()
    return np.diagonal(einsumt('ij,ik->jk', A, B)).sum()  # much faster


def compute_scaling(A, B):
    return compute_sum_inner_product(A, B) / (np.linalg.norm(B, ord='fro') ** 2)


def compute_sum_outer_product(A, B):
    # return B.T @ A
    return einsumt('ij,ik->jk', B, A)  # much faster than above


def apply_ao_rotation(A, B):
    """Apply algo 2.1: SVD-based rotation."""
    H = compute_sum_outer_product(A, B)
    U, _, VT = np.linalg.svd(H)  # decompose
    R = U.dot(VT)  # build rotation
    return B.dot(R)


def apply_absolute_orientation_with_scaling(A, B):
    """Apply algo 2.4."""
    BR = apply_ao_rotation(A, B)  # rotated B
    s = compute_scaling(A, BR)
    return s * BR


def root_mean_square_error(x, y):
    """Return root mean squared error"""
    return np.sqrt(((x - y) ** 2).mean())


def align(A, B):
    """Align two matrices with AO+scaling and return RMSE."""
    T = apply_absolute_orientation_with_scaling(A, B)
    V = apply_absolute_orientation_with_scaling(B, A)
    rmse1 = root_mean_square_error(A, T)
    rmse2 = root_mean_square_error(B, V)
    avg = (rmse1 + rmse2) / 2
    return avg


def main():
    """Launch matrixor."""
    parser = argparse.ArgumentParser(prog='matrixor')
    subparsers = parser.add_subparsers()
    parser_align = subparsers.add_parser(
        'align', formatter_class=argparse.RawTextHelpFormatter,
        help='align two vector spaces')
    parser_align.set_defaults(func=align)
    args = parser.parse_args()
    args.func(args)
