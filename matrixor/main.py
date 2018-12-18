"""Welcome to matrixor.

This is the entry point of the application.
"""

import os

import argparse
import logging
import logging.config


import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from scipy import spatial as sp

import matrixor.utils.config as cutils
import matrixor.transformation.transformator as trsfor

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def rmse(x, y):
    """Return root mean squared error"""
    return np.sqrt(((x - y) ** 2).mean())


def _get_cosine_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def _compare(args):
    logger.info('Comparing models...')
    logger.info('Loading model {}'.format(args.model_1))
    A = np.load(args.model_1)
    logger.info('Loading model {}'.format(args.model_2))
    B = np.load(args.model_2)
    vocab_filepath = '{}.vocab'.format(args.model_1.rsplit('.vec')[0])
    logger.info('Loading vocabulary from {}'.format(vocab_filepath))
    words = _load_vocab(vocab_filepath)
    logger.info('Computing similarities...')
    results_filepath = os.path.join(os.path.dirname(args.model_1),
                                    'compare.results')
    logger.info('Saving results to {}'.format(results_filepath))
    with open(results_filepath, 'w') as results_stream:
        for idx in range(A.shape[0]):
            sim = _get_cosine_sim(A[idx], B[idx])
            print('word: {:20} sim = {}'.format(words[idx], sim),
                  file=results_stream)
        print('RMSE = {}'.format(rmse(A, B)), file=results_stream)


def _load_vocab(vocab_filepath):
    with open(vocab_filepath, 'r') as vocab_stream:
        return [line.strip().split('\t')[0] for line in vocab_stream]


def _align(args):
    logger.info('Aligning input models...')
    model_1_vocab_filepath = '{}.vocab'.format(args.model_1.rsplit('.vec')[0])
    model_2_vocab_filepath = '{}.vocab'.format(args.model_2.rsplit('.vec')[0])
    logger.info('Loading model-1 vocabulary from {}'
                .format(model_1_vocab_filepath))
    vocab_1 = _load_vocab(model_1_vocab_filepath)
    logger.info('Loading model-2 vocabulary from {}'
                .format(model_2_vocab_filepath))
    vocab_2 = _load_vocab(model_2_vocab_filepath)
    if vocab_1 != vocab_2:
        raise Exception(
            'The specified models do not have the same vocabulary. Matrixor '
            'cannot align embeddings which do not have the same number of '
            'rows and columns')
    logger.info('Loading {}'.format(args.model_1))
    A = np.load(args.model_1)
    logger.info('Loading {}'.format(args.model_2))
    B = np.load(args.model_2)
    logger.info('Transforming matrices...')
    AC, RBC = trsfor.align_ao_centered(A, B)
    logger.info('Saving aligned models...')
    np.save('{}.aligned'.format(args.model_1.rsplit('.npy')[0]), AC)
    np.save('{}.aligned'.format(args.model_2.rsplit('.npy')[0]), RBC)


def _get_neighbours_idx(matrix, vector):
    v = vector.reshape(1, -1)
    sims = sp.distance.cdist(matrix, v, 'cosine').reshape(-1)
    return np.argsort(sims)


def _update_rr_and_count(relative_ranks, count, rank):
    relative_rank = 1.0 / float(rank)
    relative_ranks += relative_rank
    count += 1
    logger.info('Rank, Relative Rank = {} {}'.format(rank, relative_rank))
    logger.info('MRR = {}'.format(relative_ranks/count))
    return relative_ranks, count


def _test(args):
    logger.info('Testing input models on the specified vocab')
    results_filepath = os.path.join(os.path.dirname(args.model_1),
                                    'def.test.results')
    logger.info('Saving results to {}'.format(results_filepath))
    vocab = _load_vocab(args.vocab)
    model_vocab_filepath = '{}.vocab'.format(args.model_1.rsplit('.vec')[0])
    words = _load_vocab(model_vocab_filepath)
    rranks = 0.0
    count = 0
    logger.info('Checking MRR on definition dataset of two background models')
    logger.info('Loading model {}'.format(args.model_1))
    embeddings_1 = np.load(args.model_1)
    logger.info('Loading model {}'.format(args.model_2))
    embeddings_2 = np.load(args.model_2)
    with open(results_filepath, 'w') as results_stream:
        print('model-1: {}'.format(args.model_1), file=results_stream)
        print('model-2: {}'.format(args.model_2), file=results_stream)
        print('vocab: {}'.format(args.vocab), file=results_stream)
        for word in vocab:
            logger.info('word = {}'.format(word))
            idx = words.index(word)
            nns = _get_neighbours_idx(embeddings_1, embeddings_2[idx])
            #logger.info('10 most similar words: {}'.format([words[idx] for idx in nns][:10]))
            rank = np.where(nns==idx)[0][0] + 1
            rranks, count = _update_rr_and_count(rranks, count, rank)
            print('word: {:15} rank = {:7} MRR = {}'
                  .format(word, rank, rranks/count), file=results_stream)
        logger.info('Final MRR =  {}'.format(rranks/count))
        print('Final MRR = {}'.format(rranks/count), file=results_stream)


def _convert(args):
    vectors_filepath = '{}.vec'.format(args.model)
    vocab_filepath = '{}.vocab'.format(args.model)
    model = Word2Vec.load(args.model)
    vectors = np.empty(shape=(len(model.wv.vocab), model.vector_size), dtype=float)
    logger.info('Saving vocabulary to {}'.format(vocab_filepath))
    with open(vocab_filepath, 'w') as vocab_stream:
        idx = 0
        for item in tqdm(sorted(model.wv.vocab.items(), key=lambda x: (x[1].count, x[0]), reverse=True)):
            print('{}\t{}'.format(item[0], item[1].count), file=vocab_stream)
            vectors[idx] = model.wv.get_vector(item[0])
            idx += 1
    logger.info('Saving vectors to {}'.format(vectors_filepath))
    np.save(vectors_filepath, vectors)


def main():
    """Launch matrixor."""
    parser = argparse.ArgumentParser(prog='matrixor')
    subparsers = parser.add_subparsers()
    parser_template = argparse.ArgumentParser(add_help=False)
    parser_template.add_argument('--model-1', required=True,
                                 help='first embeddings model')
    parser_template.add_argument('--model-2', required=True,
                                 help='second embeddings model')
    parser_convert = subparsers.add_parser(
        'convert', formatter_class=argparse.RawTextHelpFormatter,
        help='convert a gensim model to matrixor .vec.npy format')
    parser_convert.set_defaults(func=_convert)
    parser_convert.add_argument('--model', required=True,
                                help='absolute path to the gensim model')
    parser_align = subparsers.add_parser(
        'align', formatter_class=argparse.RawTextHelpFormatter,
        parents=[parser_template],
        help='transform a given vector space')
    parser_align.set_defaults(func=_align)
    parser_compare = subparsers.add_parser(
        'compare', formatter_class=argparse.RawTextHelpFormatter,
        parents=[parser_template],
        help='compare two vector spaces')
    parser_compare.set_defaults(func=_compare)
    parser_test = subparsers.add_parser(
        'test', formatter_class=argparse.RawTextHelpFormatter,
        parents=[parser_template],
        help='compare two datasets on a list of words')
    parser_test.add_argument('--vocab', help='a list of words')
    parser_test.set_defaults(func=_test)
    args = parser.parse_args()
    args.func(args)
