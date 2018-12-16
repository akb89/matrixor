"""Welcome to matrixor.

This is the entry point of the application.
"""

import os

import argparse
import logging
import logging.config

from ast import literal_eval

import numpy as np

import matrixor.utils.config as cutils
import matrixor.transformation.transformator as trsfor

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def _print_dict(model, n_items_dict):
    with open('{}.test'.format(model), 'w') as test_stream:
        for key, value in sorted(n_items_dict.items()):
            print('{}\t{}'.format(key, value), file=test_stream)


def _get_dict(model):
    model_map = {}
    with open(model, 'r') as model_stream:
        for line in model_stream:
            line = line.strip()
            items = line.split('\t')
            model_map[items[0]] = items[1]
    return model_map


def _generate(args):
    dict_1 = _get_dict(args.model_1)
    dict_2 = _get_dict(args.model_2)
    n_items_dict_1 = {k: dict_1[k] for k in list(dict_1)[:args.num]}
    n_items_dict_2 = {key: dict_2[key] for key in n_items_dict_1.keys()}
    _print_dict(args.model_1, n_items_dict_1)
    _print_dict(args.model_2, n_items_dict_2)


def _save(path, words, matrix):
    with open(path, 'w') as matrix_stream:
        for idx, word in enumerate(words):
            print('{}\t{}'.format(word, np.array2string(matrix[idx],
                                                        separator=',',
                                                        max_line_width=1e10)),
                  file=matrix_stream)


def _get_line_count(model):
    count = 0
    with open(model, 'r') as model_stream:
        for line in model_stream:
            line = line.strip()
            if line:
                count += 1
    return count


def _get_vectors_dim(model):
    with open(model, 'r') as model_stream:
        for line in model_stream:
            line = line.strip()
            items = line.split('\t')
            return len(literal_eval(items[1]))
    return 0


def _load(model):
    words = []
    num_rows = _get_line_count(model)
    num_cols = _get_vectors_dim(model)
    matrix = np.empty(shape=(num_rows, num_cols), dtype=float)
    with open(model, 'r') as model_stream:
        for idx, line in enumerate(model_stream):
            line = line.strip()
            if line:
                items = line.split('\t')
                words.append(items[0])
                matrix[idx] = np.fromiter(literal_eval(items[1]), dtype=float)
    return matrix, words


def _get_cosine_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def rmse(x, y):
    """Return root mean squared error"""
    return np.sqrt(((x - y) ** 2).mean())


def _compare(args):
    A, words = _load(args.model_1)
    B, _ = _load(args.model_2)
    for idx in range(A.shape[0]):
        sim = _get_cosine_sim(A[idx], B[idx])
        print('word: {} sim = {}'.format(words[idx], sim))
    print('RMSE = {}'.format(rmse(A, B)))


def _align(args):
    A, words = _load(args.model_1)
    B, _ = _load(args.model_2)
    AC, RBC = trsfor.align_ao_centered(A, B)
    _save('{}.aligned'.format(args.model_1), words, AC)
    _save('{}.aligned'.format(args.model_2), words, RBC)


def _load_vocab(vocab_filepath):
    with open(vocab_filepath, 'r') as vocab_stream:
        return [line.strip() for line in vocab_stream]


def _load_word_vec_dict(model_filepath):
    word_vec_dict = {}
    with open(model_filepath, 'r') as model_stream:
        for line in model_stream:
            line = line.strip()
            items = line.split('\t')
            word_vec_dict[items[0]] = np.fromiter(literal_eval(items[1]),
                                                  dtype=float)
    return word_vec_dict


def _get_neighbours_by_vector(word_vec_dict, vector):
    sim_dict = {word: _get_cosine_sim(word_vector, vector)
                for word, word_vector in word_vec_dict.items()}
    return [item[0] for item in sorted(sim_dict.items(), key=lambda x: x[1],
                                       reverse=True)]


def _update_rr_and_count(relative_ranks, count, rank):
    relative_rank = 1.0 / float(rank)
    relative_ranks += relative_rank
    count += 1
    logger.info('Rank, Relative Rank = {} {}'.format(rank, relative_rank))
    logger.info('MRR = {}'.format(relative_ranks/count))
    return relative_ranks, count


def _test(args):
    vocab = _load_vocab(args.vocab)
    relative_ranks = 0.0
    count = 0
    logger.info('Checking MRR on definition dataset of two background models')
    embeddings_1 = _load_word_vec_dict(args.model_1)
    embeddings_2 = _load_word_vec_dict(args.model_2)
    for word in vocab:
        logger.info('word = {}'.format(word))
        nns = _get_neighbours_by_vector(embeddings_1, embeddings_2[word])
        logger.info('10 most similar words: {}'.format(nns[:10]))
        rank = nns.index(word) + 1
        rranks, count = _update_rr_and_count(relative_ranks, count, rank)
    logger.info('Final MRR =  {}'.format(rranks/count))


def main():
    """Launch matrixor."""
    parser = argparse.ArgumentParser(prog='matrixor')
    subparsers = parser.add_subparsers()
    parser_template = argparse.ArgumentParser(add_help=False)
    parser_template.add_argument('--model-1', required=True,
                                 help='first embeddings model')
    parser_template.add_argument('--model-2', required=True,
                                 help='second embeddings model')
    parser_generate = subparsers.add_parser(
        'generate', formatter_class=argparse.RawTextHelpFormatter,
        parents=[parser_template],
        help='generate test set with n items')
    parser_generate.set_defaults(func=_generate)
    parser_generate.add_argument('--num', type=int,
                                 help='number of test instances')
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
