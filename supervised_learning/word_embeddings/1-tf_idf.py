#!/usr/bin/env python3
import numpy as np
import re
"""

"""


def tf_idf(sentences, vocab=None):
    def preprocess_sentence(sentence):
        processed_sentence = re.sub(r"\b(\w+)'s\b", r"\1", sentence.lower())
        return re.findall(r'\w+', processed_sentence)
    if vocab is None:
        all_words = [word for sentence in sentences for word in preprocess_sentence(sentence)]
        vocab = sorted(set(all_words))
    tf_idf_scores = []
    for sentence in sentences:
        words = preprocess_sentence(sentence)
        tf_idf_sentence = []
        for word in vocab:
            word_count = sum(1 for w in words if w == word)
            total_words = len(words)
            tf = word_count / total_words if total_words > 0 else 0
            word_appears = sum(1 for sent in sentences if word in preprocess_sentence(sent))
            idf = np.log(len(sentences) / (1 + word_appears))
            tf_idf_score = tf * idf
            tf_idf_sentence.append(tf_idf_score)
        tf_idf_scores.append(tf_idf_sentence)
    return np.array(tf_idf_scores), vocab
