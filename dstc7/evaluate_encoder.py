"""Script for evaluating a PolyAI encoder model on the DSTC 7 test data.

Usage:

    python dstc7/evaluate_encoder.py \
        --examples_json ubuntu_test_subtask_1.json \
        --labels_tsv ubuntu_responses_subtask_1.tsv \
        --encoder http://models.poly-ai.com/dstc7_ubuntu_encoder/v1/model.tar.gz

Copyright PolyAI Limited.
"""  # NOQA long line

import argparse

import glog
import numpy
import tensorflow_text

import encoder_client
from dstc7 import test_reader

[tensorflow_text]  # Convince flake8 it is used.


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        help="TFhub uri for encoder model.",
        default=(
            "http://models.poly-ai.com/dstc7_ubuntu_encoder/v1/model.tar.gz"),
    )
    parser.add_argument(
        "--examples_json",
        required=True,
        help=("file name of json file containing conversational contexts and "
              "candidate responses."),
    )
    parser.add_argument(
        "--labels_tsv",
        required=True,
        help=("file name of tsv file containing labels corresponding to "
              "--examples_json."),
    )
    return parser.parse_args()


def _evaluate(client, examples):
    # Compute context encodings.
    context_encodings = client.encode_contexts(
        contexts=[example.context for example in examples],
        extra_contexts=[example.extra_contexts for example in examples],
    )

    # Iterate through examples and score candidates.
    ranks = []
    for i, example in enumerate(examples):
        responses = [example.response] + example.distractors
        response_encodings = client.encode_responses(responses)
        scores = context_encodings[i].dot(response_encodings.T)

        # Find the position of 0 in the argsort, as index 0 is the correct
        # response.
        ranks.append((-scores).argsort().argmin())
        if (i + 1) % 100 == 0:
            glog.info(f"Scored {i + 1} / {len(examples)} examples.")

    ranks = numpy.asarray(ranks)

    for k in [1, 10, 50]:
        recall_at_k = (ranks < k).mean()
        glog.info(f"Recall@{k} = {recall_at_k:.3f}")

    mrr = (1 / (1.0 + ranks)).mean()
    glog.info(f"MRR = {mrr:.3f}")


if __name__ == "__main__":
    flags = _parse_args()
    client = encoder_client.EncoderClient(
        flags.encoder,
        use_extra_context=True,
        use_extra_context_prefixes=True,
        max_extra_contexts=10,
    )
    reader = test_reader.TestReader(
        examples_json=flags.examples_json,
        labels_tsv=flags.labels_tsv,
    )
    _evaluate(client, reader.examples)
