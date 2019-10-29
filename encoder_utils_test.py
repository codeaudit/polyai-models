"""Tests for encoder_utils.py.

Copyright PolyAI Limited.
"""

import random

import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_text

import encoder_utils

# Convince flake8 the module is used. Required for loading tfhub module.
[tensorflow_text]


def _assert_tokens_equal(test_case, ragged_tokens, expected_tokens):
    tokens_list = ragged_tokens.to_list()
    tokens_list = [
        [x.decode("utf-8") for x in tokens]
        for tokens in tokens_list
    ]
    test_case.assertEqual(tokens_list, expected_tokens)


def _tokenize_function():
    # Create tf graph for subword tokenizing, using test hub module.
    module = tfhub.Module(
        "testdata/tfhub_modules/encoder",
        name="encoder",
        trainable=False,
    )
    text = tf.placeholder(dtype=tf.string, shape=[1])
    tokens = module(text, signature="tokenize")
    return text, tokens


class DetokenizeTest(tf.test.TestCase):
    """Test the detokenize method."""
    def test_token_start(self):
        """This should never change for backwards compatability."""
        self.assertEqual(encoder_utils.TOKEN_START, u"﹏")

    def test_detokenize(self):
        subtokens = ["﹏", "code", "﹏encod", "er", "﹏."]
        self.assertEqual(
            "code encoder.", encoder_utils.detokenize(subtokens))
        subtokens = ["﹏こん", "にち", "は", "﹏、", "﹏世界", "﹏。"]
        self.assertEqual(
            "こんにちは、世界。", encoder_utils.detokenize(subtokens))

    def test_tokenize_detokenize(self):
        with self.session() as sess:
            text_placeholder, tokens_tensor = _tokenize_function()
            sess.run(tf.tables_initializer())

            def tokenize(text_input):
                return [
                    token.decode("utf-8")
                    for token in sess.run(
                        tokens_tensor, {text_placeholder: [text_input]})[0]
                ]

            test_strings = [
                "hello world",
                "hello, how are you?",
                u"love: ❤️, cat: 🐈",
                ":-)",
                u"こんにちは :~)",
                u"外務省の危険情報を中心に、世界各国・",
                u"(金) 23:30(jst)ほか",
                "",
                "this sentence  has    lots of  spaces."
            ]
            for test_string in test_strings:
                self.assertEqual(
                    test_string, encoder_utils.detokenize(
                        tokenize(test_string)))

    def test_tokenize_detokenize_fuzz(self):
        with self.session() as sess:
            text_placeholder, tokens_tensor = _tokenize_function()
            sess.run(tf.tables_initializer())

            def tokenize(text_input):
                return [
                    token.decode("utf-8")
                    for token in sess.run(
                        tokens_tensor, {text_placeholder: [text_input]})[0]
                ]

            alphabet = u"abcdefg :.?!-@ éàèù 你好 ， 世界 包子"

            for _ in range(1000):
                test_string = "".join(
                    [random.choice(alphabet) for _ in range(32)])
                self.assertEqual(
                    test_string, encoder_utils.detokenize(
                        tokenize(test_string)))

    def test_detokenize_single_char_words(self):
        tokens = ["﹏can", "﹏i", "﹏book", "﹏"]
        self.assertEqual("can i book", encoder_utils.detokenize(tokens))


class SubtokenSpansTest(tf.test.TestCase):
    """Test the subtoken_spans method."""

    def _decode_bytes(self, bytes_matrix):
        """Convert a matrix of bytes to a list of lists of strings."""
        return [
            [value.decode("utf-8") for value in row]
            for row in bytes_matrix
        ]

    def test_all_alphanumeric(self):
        with self.session() as sess:
            tokens_dense = [
                ["﹏my", "﹏name", "﹏is", "﹏matt", ""],
                ["﹏i", "﹏am", "﹏matt", "h", "ew"]
            ]
            spans = sess.run(encoder_utils.subtoken_spans(tokens_dense))
            self.assertAllClose(
                spans,
                [
                    [
                        [0, 2],    # "my"
                        [2, 7],    # " name"
                        [7, 10],   # " is"
                        [10, 15],  # " Matt"
                        [15, 15],  # padding
                    ],
                    [
                        [0, 1],    # "I"
                        [1, 4],    # " am"
                        [4, 9],    # " matt"
                        [9, 10],   # "h"
                        [10, 12],  # "ew"
                    ]
                ]
            )

    def test_with_non_alphanum(self):
        with self.session() as sess:
            tokens_dense = [
                ["﹏hello", "﹏.", " ", "﹏", "h", "i", "﹏!", "", "", ""],
                ["﹏i", "﹏am", "﹏", " ", " ", " ", " ", "﹏matt", "﹏.",
                 ""],
                ["﹏.", ".", "﹏hello", "﹏?", " ", "﹏matt", "h", "ew",
                 "﹏", "h"],
            ]
            spans = sess.run(encoder_utils.subtoken_spans(tokens_dense))
            self.assertAllClose(
                spans,
                [
                    [
                        [0, 5],     # "hello"
                        [5, 6],     # "."
                        [6, 7],     # " "
                        [7, 7],     # "﹏"
                        [7, 8],     # "h"
                        [8, 9],     # "i"
                        [9, 10],    # "!"
                        [10, 10],   # ""
                        [10, 10],   # ""
                        [10, 10],   # ""
                    ],
                    [
                        [0, 1],    # "I"
                        [1, 4],    # " am"
                        [4, 4],    # "﹏"
                        [4, 5],    # " "
                        [5, 6],    # " "
                        [6, 7],    # " "
                        [7, 8],    # " "
                        [8, 12],   # "matt"
                        [12, 13],  # "."
                        [13, 13],  # ""
                    ],
                    [
                        [0, 1],    # "."
                        [1, 2],    # "."
                        [2, 7],    # "hello"
                        [7, 8],    # "?"
                        [8, 9],    # " "
                        [9, 13],   # "matt"
                        [13, 14],   # "h"
                        [14, 16],  # "ew"
                        [16, 17],  # "﹏ "
                        [17, 18],  # "h"
                    ],
                ]
            )

    def test_fuzz(self):
        """Test with random fuzzed inputs."""
        with self.session() as sess:
            text_placeholder, tokens_tensor = _tokenize_function()
            sess.run(tf.tables_initializer())
            spans = encoder_utils.subtoken_spans(tokens_tensor)

            alphabet = u"abcdefg :.?!-@ éàèù 你好 ， 世界 包子 "
            for _ in range(1000):
                test_string = "".join(
                    [random.choice(alphabet) for _ in range(32)])
                spans_value, tokens_value = sess.run(
                    (spans, tokens_tensor),
                    {text_placeholder: [test_string]})
                self.assertEqual(
                    spans_value.shape[1], tokens_value.shape[1])
                last_end_index = None
                for (start_index, end_index), token in zip(
                        spans_value[0], tokens_value[0]):
                    if last_end_index is not None:
                        self.assertEqual(start_index, last_end_index)
                    last_end_index = end_index
                    token = token.decode("utf-8")
                    substring = test_string[start_index:end_index]
                    if token.startswith(encoder_utils.TOKEN_START):
                        if substring.startswith(" "):
                            self.assertEqual(token[1:], substring[1:])
                        else:
                            self.assertEqual(token[1:], substring)
                    else:
                        self.assertEqual(token, substring)
                self.assertEqual(last_end_index, len(test_string))


if __name__ == "__main__":
    tf.test.main()
