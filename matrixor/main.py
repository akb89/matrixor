"""Welcome to matrixor.

This is the entry point of the application.
"""

import os

import argparse
import logging
import logging.config

import numpy as np

import matrixor.utils.config as cutils

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def main():
    """Launch matrixor."""
    parser = argparse.ArgumentParser(prog='matrixor')
    subparsers = parser.add_subparsers()
    # a shared set of parameters when using gensim
    parser_gensim = argparse.ArgumentParser(add_help=False)
    parser_gensim.add_argument('--num-threads', type=int, default=1,
                               help='number of threads to be used by gensim')
    parser_gensim.add_argument('--alpha', type=float,
                               help='initial learning rate')
    parser_gensim.add_argument('--neg', type=int,
                               help='number of negative samples')
    parser_gensim.add_argument('--window', type=int,
                               help='window size')
    parser_gensim.add_argument('--sample', type=float,
                               help='subsampling rate')
    parser_gensim.add_argument('--epochs', type=int,
                               help='number of epochs')
    parser_gensim.add_argument('--min-count', type=int,
                               help='min frequency count')

    # a shared set of parameters when using informativeness
    parser_info = argparse.ArgumentParser(add_help=False)
    parser_info.add_argument('--info-model', type=str,
                             help='informativeness model path')
    parser_info.add_argument('--sum-filter', default=None,
                             choices=['random', 'self', 'cwi'],
                             help='filter for sum initialization')
    parser_info.add_argument('--sum-threshold', type=int,
                             dest='sum_thresh',
                             help='sum filter threshold for self and cwi')
    parser_info.add_argument('--train-filter', default=None,
                             choices=['random', 'self', 'cwi'],
                             help='filter over training context')
    parser_info.add_argument('--train-threshold', type=int,
                             dest='train_thresh',
                             help='train filter threshold for self and cwi')
    parser_info.add_argument('--sort-by', choices=['asc', 'desc'],
                             default=None,
                             help='cwi sorting order for context items')

    # train word2vec with gensim from a wikipedia dump
    parser_train = subparsers.add_parser(
        'train', formatter_class=argparse.RawTextHelpFormatter,
        parents=[parser_gensim],
        help='generate pre-trained embeddings from wikipedia dump via '
             'gensim.word2vec')
    parser_train.set_defaults(func=_train)
    parser_train.add_argument('--data', required=True, dest='datadir',
                              help='absolute path to training data directory')
    parser_train.add_argument('--size', type=int, default=400,
                              help='vector dimensionality')
    parser_train.add_argument('--train-mode', choices=['cbow', 'skipgram'],
                              help='how to train word2vec')
    parser_train.add_argument('--outputdir', required=True,
                              help='Absolute path to outputdir to save model')
    args = parser.parse_args()
    args.func(args)
