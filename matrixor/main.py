"""Welcome to matrixor.

This is the entry point of the application.

Implements algorithm 2.4 (AO+Scaling) of the paper:

@article{devetal2018,
    title={{Absolute Orientation for Word Embedding Alignment}},
    author={Sunipa Dev and Safia Hassan and Jeff M. Phillips},
    journal={CoRR},
    year={2018},
    volume={abs/1806.01330}
}
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

__all__ = ('align', 'root_mean_square_error')


def root_mean_square_error(x, y):
    """Return root mean squared error."""
    return np.sqrt(((x - y) ** 2).mean())


def center(matrix):
    """Center input matrix."""
    return matrix - matrix.mean(axis=0)


def compute_sum_inner_product(A, B):
    """Return sum of inner product between A and B."""
    # return np.diagonal(A @ B.T).sum()
    return np.diagonal(einsumt('ij,ik->jk', A, B)).sum()  # much faster


def compute_scaling(A, B):
    """Scale matrix B to matrix A."""
    return compute_sum_inner_product(A, B) / (np.linalg.norm(B,
                                                             ord='fro') ** 2)


def compute_sum_outer_product(A, B):
    """Return sim of outer product between A and B."""
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


# pylint: disable=W1114
def align(A, B, average=True):
    """Align two matrices with AO+scaling and return RMSE."""
    T = apply_absolute_orientation_with_scaling(A, B)
    rmse1 = root_mean_square_error(A, T)
    if average:
        V = apply_absolute_orientation_with_scaling(B, A)
        rmse2 = root_mean_square_error(B, V)
        return (rmse1 + rmse2) / 2
    return rmse1


def _align(args):
    return align(args.matrix_1, args.matrix_2, args.average)


def main():
    """Launch matrixor."""
    parser = argparse.ArgumentParser(prog='matrixor')
    subparsers = parser.add_subparsers()
    parser_align = subparsers.add_parser(
        'align', formatter_class=argparse.RawTextHelpFormatter,
        help='align two numpy models by applying Absolute Orienting + Scaling'
             'and return RMSE')
    parser_align.set_defaults(func=_align)
    parser_align.add_argument('-m1', '--matrix-1', required=True,
                              help='absolute path to first .npy matrix')
    parser_align.add_argument('-m2', '--matrix-2', required=True,
                              help='absolute path to second .npy matrix')
    parser_align.add_argument('-a', '--average', action='store_true',
                              help='if set, will average RMSE between aligning'
                                   'm1, m2 and m2, m1')
    args = parser.parse_args()
    args.func(args)
