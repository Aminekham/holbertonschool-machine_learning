#!/usr/bin/env python3
import numpy as np
import re


"""

"""
def preprocess_sentence(sentence):
    """
    
    """
    processed_sentence = re.sub(r"\b(\w+)'s\b", r"\1", sentence.lower())
    return re.findall(r'\w+', processed_sentence)

def bag_of_words(sentences, vocab=None):
    """
    
    """
    preprocessed_sentences = [preprocess_sentence(sentence) for sentence in sentences]
    if vocab is None:
        all_words = [word for sentence in preprocessed_sentences for word in sentence]
        vocab = sorted(set(all_words))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    for i, sentence in enumerate(preprocessed_sentences):
        for word in sentence:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1
    return embeddings, vocab
