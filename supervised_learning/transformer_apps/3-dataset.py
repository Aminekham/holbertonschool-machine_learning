#!/usr/bin/env python3
"""

"""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

class Dataset:
    """

    """
    def __init__(self, batch_size, max_len):
        """

        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)
        self.vocab_size_pt = self.tokenizer_pt.vocab_size
        self.vocab_size_en = self.tokenizer_en.vocab_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.data_train = self.prepare_dataset(self.data_train)
        self.data_valid = self.prepare_dataset(self.data_valid, training=False)

    def prepare_dataset(self, data, training=True):
        """

        """
        def filter_max_length(pt, en):
            return tf.logical_and(tf.size(pt) <= self.max_len, tf.size(en) <= self.max_len)
        data = data.map(self.tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.filter(filter_max_length)
        if training:
            data = data.cache()
            data = data.shuffle(10000)
        data = data.padded_batch(self.batch_size, padded_shapes=([None], [None]))
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data

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

    def tf_encode(self, pt, en):
        """

        """
        pt_encoded, en_encoded = tf.py_function(func=self.encode, inp=[pt, en], Tout=[tf.int64, tf.int64])
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])        
        return pt_encoded, en_encoded
