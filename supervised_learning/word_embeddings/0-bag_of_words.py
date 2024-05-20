#!/usr/bin/env python3
import numpy as np

"""

"""
def bag_of_words(sentences, vocab=None):
    """
    
    """
    tokenized_sentences = [sentence.split() for sentence in sentences]
    if vocab is None:
        all_words = set(word for sentence in tokenized_sentences for word in sentence)
        vocab = sorted(list(all_words))
    embeddings = []
    for sentence in tokenized_sentences:
        embd = []
        for word in sentence:
            if word in vocab:
                embd.append(1)
            else:
                embd.append(0)
        embeddings.append(embd)    
    return embeddings, vocab
