"""Welcome to matrixor.

This is the entry point of the application.
"""

import os

import argparse
import logging
import logging.config

import scipy.spatial as spatial
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from scipy import spatial as sp
from scipy import stats

#from sklearn.metrics import mean_squared_error

import matrixor.utils.config as cutils
import matrixor.transformation.transformator as trsfor

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def rmse(x, y):
    """Return root mean squared error"""
    #return math.sqrt(mean_squared_error(x, y))
    #return np.linalg.norm(x - y) / np.sqrt(len(x))
    return np.sqrt(((x - y) ** 2).mean())


def _get_cosine_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def _get_nearest_neighbors(x, B, n, vocab):
    cos = 1 - spatial.distance.cdist([x], B, 'cosine')
    indices = np.argsort(cos)
    #print(indices)
    indices = indices[0][::-1]
    return [vocab[idx] for idx in indices][1:n+1]


def _compare(args):
    logger.info('Comparing models...')
    logger.info('Loading model {}'.format(args.model_1))
    A = np.load(args.model_1)
    logger.info('Loading model {}'.format(args.model_2))
    B = np.load(args.model_2)
    #vocab_filepath = '{}.vocab'.format(args.model_1.rsplit('.vec')[0])
    logger.info('Loading vocabulary from {}'.format(args.vocab))
    vocab = _load_vocab(args.vocab)
    logger.info('Computing similarities...')
    results_filepath = os.path.join(os.path.dirname(args.model_1),
                                    '{}.{}.top-{}.compare.results'
                                    .format(args.name_1, args.name_2,
                                            args.neighbors))
    logger.info('Saving results to {}'.format(results_filepath))
    avg_sim = 0
    avg_inter = 0
    n = args.neighbors
    with open(results_filepath, 'w') as results_stream:
        for idx, word in tqdm(vocab.items()):
            if A.shape == B.shape:
                sim = _get_cosine_sim(A[idx], B[idx])
                avg_sim += sim
                print('word: {:20} sim = {}'.format(word, sim),
                      file=results_stream)
            neighbors_model_1 = _get_nearest_neighbors(A[idx], A, n, vocab)
            neighbors_model_2 = _get_nearest_neighbors(B[idx], B, n, vocab)
            inter = (len(set(neighbors_model_1).intersection(neighbors_model_2)) / n) * 100
            avg_inter += inter
            print('Neigbours of "{}" in {:10}: {}'
                  .format(word, args.name_1, neighbors_model_1),
                  file=results_stream)
            print('Neigbours of "{}" in {:10}: {}'
                  .format(word, args.name_2, neighbors_model_2),
                  file=results_stream)
            print('Intersection = {}%'.format(inter), file=results_stream)
            print('RMSE = {}'.format(rmse(A, B)), file=results_stream)
        if A.shape == B.shape:
            print('Average cosine = {}'.format(avg_sim / len(vocab)),
                  file=results_stream)
        print('Average intersection on top-{} nearest neighbors = {}%'
              .format(n, avg_inter / len(vocab)), file=results_stream)


def _load_vocab(vocab_filepath):
    with open(vocab_filepath, 'r') as vocab_stream:
        return {int(line.strip().split('\t')[0]): line.strip().split('\t')[1] for line in vocab_stream}


def _align(args):
    logger.info('Aligning input models...')
    # model_1_vocab_filepath = '{}.vocab'.format(args.model_1.rsplit('.vec')[0])
    # model_2_vocab_filepath = '{}.vocab'.format(args.model_2.rsplit('.vec')[0])
    # logger.info('Loading model-1 vocabulary from {}'
    #             .format(model_1_vocab_filepath))
    # vocab_1 = _load_vocab(model_1_vocab_filepath)
    # logger.info('Loading model-2 vocabulary from {}'
    #             .format(model_2_vocab_filepath))
    # vocab_2 = _load_vocab(model_2_vocab_filepath)
    # if vocab_1 != vocab_2:
    #     raise Exception(
    #         'The specified models do not have the same vocabulary. Matrixor '
    #         'cannot align embeddings which do not have the same number of '
    #         'rows and columns')
    logger.info('Loading {}'.format(args.model_1))
    A = np.load(args.model_1)
    logger.info('Loading {}'.format(args.model_2))
    B = np.load(args.model_2)
    logger.info('Transforming matrices...')
    AC, RBC = trsfor.align_ao_centered(A, B)
    logger.info('Saving aligned models...')
    np.save('{}.aligned'.format(args.model_1.rsplit('.npy')[0]), AC)
    np.save('{}.aligned'.format(args.model_2.rsplit('.npy')[0]), RBC)


def _get_neighbors_idx(matrix, vector):
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
            nns = _get_neighbors_idx(embeddings_1, embeddings_2[idx])
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
    parser_compare.add_argument('--vocab', required=True,
                                help='absolute path to vocabulary')
    parser_compare.add_argument('-n', '--neighbors', type=int, required=True,
                                help='number of nearest neighbors to consider')
    parser_compare.add_argument('--name-1', required=True,
                                help='name of the first dataset')
    parser_compare.add_argument('--name-2', required=True,
                                help='name of the second dataset')
    parser_test = subparsers.add_parser(
        'test', formatter_class=argparse.RawTextHelpFormatter,
        parents=[parser_template],
        help='compare two datasets on a list of words')
    parser_test.add_argument('--vocab', help='a list of words')
    parser_test.set_defaults(func=_test)
    args = parser.parse_args()
    args.func(args)
