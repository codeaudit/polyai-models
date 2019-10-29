"""Utils for encoder models.

Copyright PolyAI Limited.
"""

import regex as re
import tensorflow as tf

# Special character used to denote the start of a token.
TOKEN_START = u"﹏"


def detokenize(tokens):
    """Detokenize a list of subword tokens."""
    joined = "".join(tokens)
    detokenized = ""
    prev_is_alnum = False

    for i, char in enumerate(joined[:-1]):
        next_char = joined[i + 1]
        if char == TOKEN_START and prev_is_alnum and _isalnum(next_char):
            # Add spaces for token start boundaries between alphanumeric
            # words.
            detokenized += " "
        elif char != TOKEN_START:
            detokenized += char

        prev_is_alnum = _isalnum(char)

    if len(joined) > 0:
        if joined[-1] != TOKEN_START:
            detokenized += joined[-1]

    return detokenized


def _isalnum(char):
    """Whether a character is considered alphanumeric.

    Consistent with `invertible_tokenize`.
    """
    return (re.match(r"[^\p{P}\p{Z}]", char) is not None)


def subtoken_spans(tokens, name=None):
    """Compute the string spans for the given subtoken tokenization.

    Args:
        tokens: a tensorflow string matrix of subtokens, where each row is a
            subword tokenization of a different string, computed using an
            encoder model. Rows should be padded with empty strings.
        name: an optional name for this op.

    Returns:
        a rank-3 tensor giving the spans of each token in each original string.
        `spans[i, j, 0]` is the start index and `spans[i, j, 1]` the end index
        of `tokens[i, j]` in the original `i`th string.
    """
    with tf.name_scope(name, default_name="subword_token_spans"):
        is_alnum = tf.strings.regex_full_match(
            tokens, r"(.*)[^\p{P}\p{Z}]$")
        starts_with_mark = tf.strings.regex_full_match(
            tokens, r"^" + TOKEN_START + r"(.*)")
        is_mark = tf.equal(tokens, TOKEN_START.encode("utf-8"))

        # We need to figure out where spaces were replaced with the
        # token start mark. This happens when the previous and next
        # subtokens are both alphanumeric.
        false_column = tf.fill([tf.shape(tokens)[0], 1], False)
        next_is_alnum = tf.concat([is_alnum[:, 1:], false_column], 1)
        prev_is_alnum = tf.concat([false_column, is_alnum[:, :-1]], 1)
        prefixed_with_space = tf.logical_or(
            tf.logical_and(is_mark,
                           tf.logical_and(prev_is_alnum, next_is_alnum)),
            tf.logical_and(tf.logical_and(starts_with_mark, is_alnum),
                           prev_is_alnum)
        )

        token_lengths = tf.strings.length(
            tf.strings.regex_replace(tokens, TOKEN_START, ""),
            unit="UTF8_CHAR",
        ) + tf.to_int32(prefixed_with_space)
        start_indices = tf.cumsum(token_lengths, axis=1, exclusive=True)
        end_indices = tf.cumsum(token_lengths, axis=1, exclusive=False)
        return tf.transpose(tf.stack([start_indices, end_indices]), [1, 2, 0])
