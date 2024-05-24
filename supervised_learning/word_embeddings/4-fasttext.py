#!/usr/bin/env python3
"""

"""
from gensim.models import FastText

def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5, cbow=True, iterations=5, seed=0, workers=1):
    sg = 1
    if cbow:
        sg = 0
    model = FastText(
        sentences=sentences, vector_size=size,
        window=window, min_count=min_count,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers,
        epochs=iterations
    )
    return model
