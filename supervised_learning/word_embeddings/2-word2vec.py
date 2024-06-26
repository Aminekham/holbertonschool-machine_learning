#!/usr/bin/env python3
from gensim.models import Word2Vec
"""

"""


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """

    """
    sg = 0
    if cbow:
        sg = 1
    model = Word2Vec(sentences, size=size,
                     min_count=min_count, window=window,
                     negative=negative, sg=sg, iter=iterations,
                     seed=seed, workers=workers)
    model.train(sentences=sentences,
                total_examples=model.corpus_count,
                epochs=iterations)
    return model
