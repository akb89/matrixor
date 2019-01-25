import numpy as np

if __name__ == '__main__':
    PATH = '/Users/AKB/GitHub/nonce2vec/models/enwiki.cbow.model.wv.vectors.npy'
    PATH2 = '/Users/AKB/GitHub/nonce2vec/models/enwiki.cbow.model'
    arr = np.load(PATH2)
    print(arr)
