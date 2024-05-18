#!/usr/bin/env python3
"""

"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    
    """
    tokenized_sentences = [sentence.split() for sentence in sentences]
    if vocab is None:
        all_words = set(word for sentence in tokenized_sentences for word in sentence)
        vocab = sorted(list(all_words))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    for i, sentence in enumerate(tokenized_sentences):
        word_counts = {}
        for word in sentence:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        for word, count in word_counts.items():
            if word in word_to_index:
                embeddings[i, word_to_index[word]] = count
    return embeddings, vocab
