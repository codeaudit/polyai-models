"""Unit tests for encoder_client.py.

Copyright PolyAI Limited.
"""

import unittest
from unittest import mock
from unittest.mock import patch

import numpy as np
import tensorflow as tf

import encoder_client


def _random_encoding(examples):
    """A random encoding function used for tests."""
    return np.random.normal(size=(len(examples), 7))


class CacheEncodingsTest(unittest.TestCase):
    def test_cache_encodings(self):
        """Test that values are cached."""
        cached_random_encoding = encoder_client.cache_encodings(
            _random_encoding, cache_size=100)
        encodings_1 = cached_random_encoding(
            ["hello", "hello world"]
        )
        self.assertEqual([2, 7], list(encodings_1.shape))
        encodings_2 = cached_random_encoding(
            ["hello world", "new input"]
        )
        self.assertEqual([2, 7], list(encodings_2.shape))

        # The encoding for "hello world" should be cached, and return the
        # same value, even though it is generated randomly.
        np.testing.assert_allclose(encodings_1[1], encodings_2[0])
        self.assertEqual(1, cached_random_encoding.cache_hits())

    def test_cache_duplicate_inputs(self):
        """Test inputs are deduplicated for the encoding function."""
        random_encoding = mock.Mock(side_effect=_random_encoding)
        cached_random_encoding = encoder_client.cache_encodings(
            random_encoding, cache_size=100)
        encodings = cached_random_encoding(["hello"] * 10)
        self.assertEqual([10, 7], list(encodings.shape))
        for i in range(1, 10):
            np.testing.assert_allclose(encodings[0], encodings[i])
        random_encoding.assert_called_once_with(["hello"])

    def test_least_recently_used_forgotten(self):
        """Test the least recently used input is forgotten."""
        cached_random_encoding = encoder_client.cache_encodings(
            _random_encoding, cache_size=10)
        encoding = cached_random_encoding(["to be forgotten"])
        cached_random_encoding(list(range(10)))
        encoding_1 = cached_random_encoding(["to be forgotten"])

        # Check the two encodings are different, as the old one should have
        # been forgotten.
        np.testing.assert_raises(
            AssertionError, np.testing.assert_allclose, encoding,
            encoding_1)
        self.assertEqual(0, cached_random_encoding.cache_hits())

    def test_nested_lists(self):
        """Test that inputs with nested lists are cached correctly."""
        cached_random_encoding = encoder_client.cache_encodings(
            _random_encoding, cache_size=100)
        example_1 = ["hello", ["context 1", "context 2"]]
        example_2 = ["hi", ["context 1", "context 2", "context 3"]]
        encodings_1 = cached_random_encoding([example_1, example_2])
        encodings_2 = cached_random_encoding([example_2, example_1])
        np.testing.assert_allclose(encodings_1[0], encodings_2[1])
        np.testing.assert_allclose(encodings_1[1], encodings_2[0])
        self.assertEqual(2, cached_random_encoding.cache_hits())


class EncoderClientTest(unittest.TestCase):
    """Test EncoderClient with a non-contextual encoder."""

    @patch("tensorflow_hub.Module")
    def test_encode_context(self, mock_module_cls):

        def mock_fn(input, signature=None):
            self.assertIn(
                signature, {"encode_context", "encode_response", None})
            self.assertIsInstance(input, tf.Tensor)
            self.assertEqual(input.dtype, tf.string)
            if signature is "encode_context":
                return tf.ones([tf.shape(input)[0], 3])

        mock_module_cls.return_value = mock_fn

        client = encoder_client.EncoderClient("test_uri")
        mock_module_cls.assert_called_with("test_uri")

        encodings = client.encode_contexts(["hello"])
        np.testing.assert_allclose([[1, 1, 1]], encodings)

    @patch("tensorflow_hub.Module")
    def test_encode_response(self, mock_module_cls):
        def mock_fn(input, signature=None):
            self.assertIn(
                signature, {"encode_context", "encode_response", None})
            self.assertIsInstance(input, tf.Tensor)
            self.assertEqual(input.dtype, tf.string)
            if signature == "encode_response":
                return tf.ones([tf.shape(input)[0], 3])

        mock_module_cls.return_value = mock_fn

        client = encoder_client.EncoderClient("test_uri")
        mock_module_cls.assert_called_with("test_uri")

        encodings = client.encode_responses(["hello"])
        np.testing.assert_allclose([[1, 1, 1]], encodings)

    @patch("tensorflow_hub.Module")
    def test_encode_sentences(self, mock_module_cls):
        def mock_fn(input, signature=None):
            self.assertIn(
                signature, {"encode_context", "encode_response", None})
            self.assertIsInstance(input, tf.Tensor)
            self.assertEqual(input.dtype, tf.string)
            if signature is None:
                return tf.ones([tf.shape(input)[0], 3])

        mock_module_cls.return_value = mock_fn

        client = encoder_client.EncoderClient("test_uri")
        mock_module_cls.assert_called_with("test_uri")

        encodings = client.encode_sentences(["hello"])
        np.testing.assert_allclose([[1, 1, 1]], encodings)

    @patch("tensorflow_hub.Module")
    def test_encode_sentences_batching_caching(self, mock_module_cls):
        def mock_fn(input, signature=None):
            self.assertIn(
                signature, {"encode_context", "encode_response", None})
            self.assertIsInstance(input, tf.Tensor)
            self.assertEqual(input.dtype, tf.string)
            if signature is None:
                return tf.random_normal([tf.shape(input)[0], 3])

        mock_module_cls.return_value = mock_fn

        client = encoder_client.EncoderClient(
            # force batching by setting batch size to 3
            "test_uri", internal_batch_size=3, cache_size=100,
        )
        mock_module_cls.assert_called_with("test_uri")

        encodings = client.encode_sentences(
            ["a", "a", "b", "c", "d", "e", "f", "g"]
        )
        # Test de-duplication:
        np.testing.assert_allclose(encodings[0], encodings[1])

        encodings_2 = client.encode_sentences(["a", "b", "c", "z"])
        # Test caching
        np.testing.assert_allclose(encodings[0], encodings_2[0])
        np.testing.assert_allclose(encodings[2], encodings_2[1])
        np.testing.assert_allclose(encodings[3], encodings_2[2])


class EncoderClientExtraContextsTest(unittest.TestCase):
    """Test EncoderClient with a contextual encoder."""

    @patch("tensorflow_hub.Module")
    def test_encode_context(self, mock_module_cls):

        def mock_fn(input, signature=None):
            self.assertIn(
                signature, {"encode_context", "encode_response", None})
            if signature == "encode_context":
                self.assertIsInstance(input, dict)
                self.assertEqual(2, len(input))
                for input_t in input.values():
                    self.assertEqual(input_t.dtype, tf.string)

                return tf.ones([tf.shape(input_t)[0], 3])

        mock_module_cls.return_value = mock_fn

        client = encoder_client.EncoderClient(
            "test_uri", use_extra_context=True)
        mock_module_cls.assert_called_with("test_uri")

        encodings = client.encode_contexts(
            ["hello", "hi", "yo"],
            extra_contexts=[
                ["a", "b", "c", "d"],
                ["A", "B", "C", "D", "E", "F"],
                [],
            ],
        )
        np.testing.assert_allclose([[1., 1., 1.]] * 3, encodings)

    @patch("tensorflow_hub.Module")
    def test_encode_context_feature_values(self, mock_module_cls):

        def mock_fn(input, signature=None):
            self.assertIn(
                signature, {"encode_context", "encode_response", None})
            if signature == "encode_context":
                self.assertIsInstance(input, dict)
                self.assertEqual(2, len(input))
                for input_t in input.values():
                    self.assertEqual(input_t.dtype, tf.string)

        mock_module_cls.return_value = mock_fn
        with mock.patch("encoder_client._batch_session_run") as f:
            client = encoder_client.EncoderClient(
                "test_uri", use_extra_context=True, max_extra_contexts=4,
                cache_size=0)
            mock_module_cls.assert_called_with("test_uri")
            encodings = client.encode_contexts(
                ["hello", "hi", "yo"],
                extra_contexts=[
                    ["a", "b", "c", "d"],
                    ["A", "B", "C", "D", "E", "F"],
                    [],
                ],
            )
            f.assert_called_once()
            self.assertEqual(
                ["d c b a", "F E D C", ""],
                list(f.call_args[0][1][client._fed_extra_contexts]),
            )
            self.assertEqual(f.return_value, encodings)

    @patch("tensorflow_hub.Module")
    def test_encode_context_feature_values_with_prefix(self, mock_module_cls):

        def mock_fn(input, signature=None):
            self.assertIn(
                signature, {"encode_context", "encode_response", None})
            if signature == "encode_context":
                self.assertIsInstance(input, dict)
                self.assertEqual(2, len(input))
                for input_t in input.values():
                    self.assertEqual(input_t.dtype, tf.string)

        mock_module_cls.return_value = mock_fn
        with mock.patch("encoder_client._batch_session_run") as f:
            client = encoder_client.EncoderClient(
                "test_uri", use_extra_context=True, max_extra_contexts=3,
                use_extra_context_prefixes=True, cache_size=0,)
            mock_module_cls.assert_called_with("test_uri")
            encodings = client.encode_contexts(
                ["hello", "hi", "yo"],
                extra_contexts=[
                    ["a", "b", "c", "d d"],
                    ["A", "B", "C", "D", "E", "F"],
                    [],
                ],
            )
            f.assert_called_once()
            self.assertEqual(
                ["0: d d 1: c 2: b", "0: F 1: E 2: D", ""],
                list(f.call_args[0][1][client._fed_extra_contexts]),
            )
            self.assertEqual(f.return_value, encodings)


if __name__ == "__main__":
    unittest.main()
