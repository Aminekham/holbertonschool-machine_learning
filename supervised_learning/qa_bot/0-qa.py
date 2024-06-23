import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
"""
question answering using bert
embeddings and the simularity between
the question and the context reference document
given by the user
"""


def question_answer(question, reference):
    """
    tokenizing both the question and the refernece
    document and choose the closest answer to that question
    based on the encodings we got using the bert tokenizer
    """
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    reference = tokenizer.tokenize(reference)
    reference = reference + ['[SEP]']
    reference_ids = tokenizer.convert_tokens_to_ids(reference)
    question = tokenizer.tokenize(question)
    question = ['[CLS]'] + question + ['[SEP]']
    question_tokens_ids = tokenizer.convert_tokens_to_ids(question)
    input_ids = question_tokens_ids + reference_ids
    input_mask = [1] * len(input_ids)
    input_types = [0] * len(question) + [1] * len(reference)
    input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    input_mask = tf.convert_to_tensor(input_mask, dtype=tf.int32)
    input_types = tf.convert_to_tensor(input_types, dtype=tf.int32)
    input_ids = tf.expand_dims(input_ids, axis=0)
    input_mask = tf.expand_dims(input_mask, axis=0)
    input_types = tf.expand_dims(input_types, axis=0)
    outputs = model([input_ids, input_mask, input_types])
    short_start = tf.argmax(outputs[0][0][1:-1]) + 1
    short_end = tf.argmax(outputs[1][0][1:-1]) + 1
    tokens = question + reference
    answer_tokens = tokens[short_start: short_end + 1]
    if len(answer_tokens) == 0:
        answer = None
    else:
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer
