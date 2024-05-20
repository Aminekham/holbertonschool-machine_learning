#!/usr/bin/env python3
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(sentences, vocab=None):
    if vocab is not None:
        vectorizer = CountVectorizer(vocabulary=vocab)
    else:
        vectorizer = CountVectorizer()
    
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    features = vectorizer.get_feature_names_out()
    
    return embeddings, features
