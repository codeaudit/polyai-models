"""Keras wrappers around encoder model tf hub modules.

Copyright PolyAI Limited.
"""

from enum import Enum

import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_text

# Convince flake8 that tensorflow_text is used. The import is required for
# defining the WordpieceTokenizer ops.
[tensorflow_text]


class EncoderLayerBase(tf.layers.Layer):
    """Base keras wrapper around the Encoder tfhub module.

    Each layer defines the subgraphs of the full model graph that it uses. This
    ensures the layer declares the correct variables as trainable, and adds
    the correct regularization losses.

    Args:
        tfhub_module: (str) location of the tfhub module.
        regularizer: (float) the multiplier for regularization losses.
    """

    def __init__(self, tfhub_module, regularizer=1.0, **kwargs):
        """Create an EncoderLayer."""
        self._tfhub_module = tfhub_module
        self._regularizer = regularizer
        super(EncoderLayerBase, self).__init__(**kwargs)

    @property
    def used_subgraphs(self):
        """The `Subgraph`s that this layer uses."""
        raise NotImplementedError("Sub-classes must implement this method.")

    def build(self, input_shape):
        """Build the layer."""
        num_regularization_losses = len(
            tf.losses.get_regularization_losses())
        self._encoder = tfhub.Module(
            self._tfhub_module, name="{}_tfhub_module".format(self.name),
            trainable=self.trainable
        )
        scope = "{}/{}_tfhub_module/".format(
            tf.get_variable_scope().name,
            self.name,
        )
        if self._regularizer > 0.0:
            new_losses = tf.losses.get_regularization_losses()[
                num_regularization_losses:]
            for loss in new_losses:
                subgraph = Subgraph.from_tensor_name(
                    _strip_scope_from_name(scope, loss.name))
                if subgraph in self.used_subgraphs:
                    inputs = None
                    if subgraph in {
                            Subgraph.CONTEXT_EMBEDDING,
                            Subgraph.EXTRA_CONTEXT_EMBEDDING,
                            Subgraph.RESPONSE_EMBEDDING}:
                        # The embedding regularizers are a function of the
                        # inputs, as they regularize the activation.
                        inputs = True
                    self.add_loss(self._regularizer * loss, inputs)
            tf.logging.info(
                "Layer adds %i regularization losses", len(self.losses))

        for variable in self._encoder.variables:
            subgraph = Subgraph.from_tensor_name(
                _strip_scope_from_name(scope, variable.name))
            if subgraph in self.used_subgraphs:
                self._trainable_weights.append(variable)
            else:
                self._non_trainable_weights.append(variable)

        super(EncoderLayerBase, self).build(input_shape)

    def call(self, sentences):
        """Compute the output of the layer."""
        raise NotImplementedError("Sub-classes must implement this.")


def _strip_scope_from_name(scope, tensor_name):
    if tensor_name.startswith(scope):
        return tensor_name[len(scope):]
    return tensor_name


class Subgraph(Enum):
    """Subgraphs of the encoder model."""
    CONTEXT_EMBEDDING = 1
    EXTRA_CONTEXT_EMBEDDING = 2
    RESPONSE_EMBEDDING = 3
    SHARED_ENCODING = 4
    CONTEXT_ENCODING = 5
    EXTRA_CONTEXT_ENCODING = 6
    COMBINED_CONTEXT_ENCODING = 7
    RESPONSE_ENCODING = 8
    SHARED_EMBEDDING = 9
    SHARED_EMBEDDING_REDUCTION = 10

    @staticmethod
    def from_tensor_name(tensor_name):
        """Infer the Subgraph of a tensor from its name."""

        if tensor_name.startswith("embed_context"):
            return Subgraph.CONTEXT_EMBEDDING
        elif tensor_name.startswith("embed_extra_context"):
            return Subgraph.EXTRA_CONTEXT_EMBEDDING
        elif tensor_name.startswith("embed_response"):
            return Subgraph.RESPONSE_EMBEDDING
        elif tensor_name.startswith("skip_connection"):
            return Subgraph.SHARED_ENCODING
        elif tensor_name.startswith("encode_context"):
            return Subgraph.CONTEXT_ENCODING
        elif tensor_name.startswith("encode_extra_context"):
            return Subgraph.EXTRA_CONTEXT_ENCODING
        elif tensor_name.startswith("encode_combined_context"):
            return Subgraph.COMBINED_CONTEXT_ENCODING
        elif tensor_name.startswith("encode_nl_feature"):
            return Subgraph.RESPONSE_ENCODING

        # It must be part of the shared embedding graph.
        if (not tensor_name.startswith("embedding_matrices") and
                not tensor_name.startswith("unigram") and
                not tensor_name.startswith("embed_nl")):
            raise ValueError("Unexpected tensor name " + tensor_name)
        if "reduction" in tensor_name:
            return Subgraph.SHARED_EMBEDDING_REDUCTION
        return Subgraph.SHARED_EMBEDDING


class SentenceEncoderLayer(EncoderLayerBase):
    """Layer that encodes sentences using the shared sentence encoding."""

    @property
    def used_subgraphs(self):
        """The `Subgraph`s that this layer uses."""
        return {
            Subgraph.CONTEXT_EMBEDDING,
            Subgraph.SHARED_EMBEDDING,
            Subgraph.SHARED_EMBEDDING_REDUCTION,
        }

    def call(self, sentences):
        """Compute the output of the layer."""
        return self._encoder(sentences)  # this is the default signature


class ContextEncoderLayer(EncoderLayerBase):
    """Layer that encodes sentences using the context encoding."""

    @property
    def used_subgraphs(self):
        """The `Subgraph`s that this layer uses."""
        return {
            Subgraph.CONTEXT_EMBEDDING,
            Subgraph.SHARED_EMBEDDING,
            Subgraph.SHARED_EMBEDDING_REDUCTION,
            Subgraph.CONTEXT_ENCODING,
            Subgraph.SHARED_ENCODING,
        }

    def call(self, sentences):
        """Compute the output of the layer."""
        return self._encoder(sentences, signature="encode_context")


class ResponseEncoderLayer(EncoderLayerBase):
    """Layer that encodes sentences using the context encoding."""

    @property
    def used_subgraphs(self):
        """The `Subgraph`s that this layer uses."""
        return {
            Subgraph.RESPONSE_EMBEDDING,
            Subgraph.SHARED_EMBEDDING,
            Subgraph.SHARED_EMBEDDING_REDUCTION,
            Subgraph.RESPONSE_ENCODING,
            Subgraph.SHARED_ENCODING,
        }

    def call(self, sentences):
        """Compute the output of the layer."""
        return self._encoder(sentences, signature="encode_response")


class ContextAndResponseEncoderLayer(EncoderLayerBase):
    """Layer that encodes contexts and responses in parallel."""

    def __init__(self, *args, uses_extra_context=False, **kwargs):
        """Create an ContextAndResponseEncoderLayer."""
        super(ContextAndResponseEncoderLayer, self).__init__(
            *args, **kwargs)
        self._uses_extra_context = uses_extra_context

    @property
    def used_subgraphs(self):
        """The `Subgraph`s that this layer uses."""
        if self._uses_extra_context:
            return list(Subgraph)
        else:
            return {
                Subgraph.CONTEXT_EMBEDDING,
                Subgraph.RESPONSE_EMBEDDING,
                Subgraph.SHARED_EMBEDDING,
                Subgraph.SHARED_EMBEDDING_REDUCTION,
                Subgraph.CONTEXT_ENCODING,
                Subgraph.RESPONSE_ENCODING,
                Subgraph.SHARED_ENCODING,
            }

    def call(self, context_and_response):
        """Compute the context and response encodings.

        The model is trained to give a high dot product between the encoding of
        a context and the encoding of an appropriate response.

        Args:
            context_and_response: the contexts and responses to encode.

        Returns:
            context_encoding: the encoded contexts.
            response_encoding: the encoded responses.
        """
        if self._uses_extra_context:
            context, extra_context, response = context_and_response
            inputs = {
                'context': context, 'response': response,
                'extra_context': extra_context,
            }
        else:
            context, response = context_and_response
            inputs = {
                'context': context, 'response': response,
            }

        outputs = self._encoder(
            inputs, signature="encode_context_and_response", as_dict=True)
        self._encoding_dim = int(outputs['context_encoding'].shape[1])
        return outputs['context_encoding'], outputs['response_encoding']

    def compute_output_shape(self, input_shape):
        """Compute the output shapes."""
        return [
            (None, self._encoding_dim),
            (None, self._encoding_dim),
        ]


class ContextualizedSubwordsLayer(EncoderLayerBase):
    """Layer that encodes text as context-aware subword embeddings"""

    @property
    def used_subgraphs(self):
        """The `Subgraph`s that this layer uses."""
        return {
            Subgraph.CONTEXT_EMBEDDING,
            Subgraph.SHARED_EMBEDDING
        }

    def call(self, sentences):
        """Compute the tokens and sequence encodings."""
        output = self._encoder(
            sentences,
            signature="encode_sequence",
            as_dict=True
        )
        return output['tokens'], output['sequence_encoding']
