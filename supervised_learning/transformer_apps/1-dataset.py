#!/usr/bin/env python3
"""

"""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

class Dataset:
    """

    """
    def __init__(self):
        """

        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)
        self.vocab_size_pt = self.tokenizer_pt.vocab_size
        self.vocab_size_en = self.tokenizer_en.vocab_size

    def tokenize_dataset(self, data):
        """

        """
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data), target_vocab_size=2**15
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data), target_vocab_size=2**15
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        
        """
        pt_tokens = [self.vocab_size_pt] + self.tokenizer_pt.encode(pt.numpy()) + [self.vocab_size_pt + 1]
        en_tokens = [self.vocab_size_en] + self.tokenizer_en.encode(en.numpy()) + [self.vocab_size_en + 1]
        return np.array(pt_tokens), np.array(en_tokens)
