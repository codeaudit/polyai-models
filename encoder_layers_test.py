"""Unit tests for encoder_layers.py.

Copyright PolyAI Limited.
"""

import tensorflow as tf

import encoder_layers

_TEST_ENCODER = "testdata/tfhub_modules/encoder"
_TEST_EXTRA_CONTEXT_ENCODER = "testdata/tfhub_modules/extra_context_encoder"


class EncoderLayersTest(tf.test.TestCase):
    def test_encode_sentences(self):
        with self.test_session() as sess:
            layer = encoder_layers.SentenceEncoderLayer(_TEST_ENCODER)
            encodings = layer(
                ["hello world", "what's up?", "hello world",
                 "sentence 4"])
            weights = [
                var for var in layer.trainable_variables
                if "layer_norm" not in var.name
            ]
            self.assertEqual(len(weights), len(layer.losses))
            sess.run([
                tf.compat.v1.local_variables_initializer(),
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.tables_initializer(),
            ])
            encodings_val = sess.run(encodings)
            self.assertEqual(list(encodings_val.shape), [4, 3])
            self.assertAllClose(encodings_val[0], encodings_val[2])
            grads = tf.gradients(
                [encodings] + layer.losses, layer.trainable_variables)
            for grad in grads:
                self.assertIsNotNone(grad)

            non_grads = tf.gradients(
                [encodings] + layer.losses, layer.non_trainable_variables)
            for grad in non_grads:
                self.assertIsNone(grad)

    def test_encode_contexts(self):
        with self.test_session() as sess:
            layer = encoder_layers.ContextEncoderLayer(_TEST_ENCODER)
            encodings = layer(
                ["hello world", "what's up?", "hello world",
                 "sentence 4"])
            weights = [
                var for var in layer.trainable_variables
                if "layer_norm" not in var.name
            ]
            self.assertEqual(len(weights), len(layer.losses))
            sess.run([
                tf.compat.v1.local_variables_initializer(),
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.tables_initializer(),
            ])
            encodings_val = sess.run(encodings)
            self.assertEqual(list(encodings_val.shape), [4, 5])
            self.assertAllClose(encodings_val[0], encodings_val[2])
            grads = tf.gradients(
                [encodings] + layer.losses, layer.trainable_variables)
            for grad in grads:
                self.assertIsNotNone(grad)

            non_grads = tf.gradients(
                [encodings] + layer.losses, layer.non_trainable_variables)
            for grad in non_grads:
                self.assertIsNone(grad)

    def test_encode_responses(self):
        with self.test_session() as sess:
            layer = encoder_layers.ResponseEncoderLayer(_TEST_ENCODER)
            encodings = layer(
                ["hello world", "what's up?", "hello world",
                 "sentence 4"])
            weights = [
                var for var in layer.trainable_variables
                if "layer_norm" not in var.name
            ]
            self.assertEqual(len(weights), len(layer.losses))
            sess.run([
                tf.compat.v1.local_variables_initializer(),
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.tables_initializer(),
            ])
            encodings_val = sess.run(encodings)
            self.assertEqual(list(encodings_val.shape), [4, 5])
            self.assertAllClose(encodings_val[0], encodings_val[2])
            grads = tf.gradients(
                [encodings] + layer.losses, layer.trainable_variables)
            for grad in grads:
                self.assertIsNotNone(grad)

            non_grads = tf.gradients(
                [encodings] + layer.losses, layer.non_trainable_variables)
            for grad in non_grads:
                self.assertIsNone(grad)

    def test_encode_contexts_and_responses(self):
        with self.test_session() as sess:
            layer = encoder_layers.ContextAndResponseEncoderLayer(
                _TEST_ENCODER)
            context_encodings, response_encodings = layer([
                ["context 1", "context 2"],
                ["response 1", "response 2", "response 3"],
            ])
            weights = [
                var for var in layer.trainable_variables
                if "layer_norm" not in var.name
            ]
            # Plus one because the embedding regularization is applied for
            # both context and response.
            self.assertEqual(len(weights) + 1, len(layer.losses))
            sess.run([
                tf.compat.v1.local_variables_initializer(),
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.tables_initializer(),
            ])
            context_encodings_val = sess.run(context_encodings)
            self.assertEqual(list(context_encodings_val.shape), [2, 5])
            response_encodings_val = sess.run(response_encodings)
            self.assertEqual(list(response_encodings_val.shape), [3, 5])

            grads = tf.gradients(
                [context_encodings, response_encodings] + layer.losses,
                layer.trainable_variables)
            for grad in grads:
                self.assertIsNotNone(grad)

            non_grads = tf.gradients(
                [context_encodings, response_encodings] + layer.losses,
                layer.non_trainable_variables)
            for grad in non_grads:
                self.assertIsNone(grad)

    def test_encode_contexts_and_responses_with_extra_contexts(self):
        with self.test_session() as sess:
            layer = encoder_layers.ContextAndResponseEncoderLayer(
                _TEST_EXTRA_CONTEXT_ENCODER, uses_extra_context=True)
            context_encodings, response_encodings = layer([
                ["context 1", "context 2"],
                ["extra context 1", "extra context 2"],
                ["response 1", "response 2", "response 3"],
            ])
            weights = [
                var for var in layer.trainable_variables
                if "layer_norm" not in var.name
            ]
            # Plus two because the embedding regularization is applied for
            # context, extra contexts, and response.
            self.assertEqual(len(weights) + 2, len(layer.losses))
            sess.run([
                tf.compat.v1.local_variables_initializer(),
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.tables_initializer(),
            ])
            context_encodings_val = sess.run(context_encodings)
            self.assertEqual(list(context_encodings_val.shape), [2, 5])
            response_encodings_val = sess.run(response_encodings)
            self.assertEqual(list(response_encodings_val.shape), [3, 5])

            grads = tf.gradients(
                [context_encodings, response_encodings] + layer.losses,
                layer.trainable_variables)
            for grad in grads:
                self.assertIsNotNone(grad)

            non_grads = tf.gradients(
                [context_encodings, response_encodings] + layer.losses,
                layer.non_trainable_variables)
            for grad in non_grads:
                self.assertIsNone(grad)

    def test_encode_to_contextualized_subwords(self):
        with self.test_session() as sess:
            layer = encoder_layers.ContextualizedSubwordsLayer(_TEST_ENCODER)
            tokens, sequence_encodings = layer(
                ["contextualised subword sequence 1", "sequence encoding 2"]
            )
            weights = [
                var for var in layer.trainable_variables
                if "layer_norm" not in var.name
            ]
            self.assertEqual(len(weights), len(layer.losses))
            sess.run([
                tf.compat.v1.local_variables_initializer(),
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.tables_initializer(),
            ])
            tokens_val = sess.run(tokens)
            self.assertEqual(list(tokens_val.shape), [2, 26])
            sequence_encodings_val = sess.run(sequence_encodings)
            self.assertEqual(list(sequence_encodings_val.shape), [2, 26, 3])

            grads = tf.gradients(
                [sequence_encodings] + layer.losses,
                layer.trainable_variables)
            for grad in grads:
                self.assertIsNotNone(grad)

            non_grads = tf.gradients(
                [sequence_encodings] + layer.losses,
                layer.non_trainable_variables)
            for grad in non_grads:
                self.assertIsNone(grad)

    def test_non_trainable(self):
        with self.test_session() as sess:
            layer = encoder_layers.ContextualizedSubwordsLayer(
                _TEST_ENCODER, trainable=False)
            tokens, sequence_encodings = layer(
                ["contextualised subword sequence 1", "sequence encoding 2"]
            )
            self.assertEqual(layer.trainable_variables, [])
            self.assertEqual(layer.losses, [])

            # check layer still works
            sess.run([
                tf.compat.v1.local_variables_initializer(),
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.tables_initializer(),
            ])
            tokens_val = sess.run(tokens)
            self.assertEqual(list(tokens_val.shape), [2, 26])
            sequence_encodings_val = sess.run(sequence_encodings)
            self.assertEqual(list(sequence_encodings_val.shape), [2, 26, 3])


if __name__ == "__main__":
    tf.test.main()
