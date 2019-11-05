"""Utils for reading the DSTC 7 Ubuntu data.

For a description of the data format, see:
    https://ibm.github.io/dstc-noesis/public/data_description.html

Copyright PolyAI Limited.
"""

import csv
import json
from collections import namedtuple


class TestReader:
    """Class for reading DSTC 7 Ubuntu test data.

    Args:
        examples_json: file name of json file containing conversational
            contexts and candidate responses.
        labels_tsv: file name of tsv file containing labels corresponding to
            `examples_json`.

    """
    def __init__(
        self,
        examples_json,
        labels_tsv,
    ):
        """Create a new TestReader."""
        self._read_data(examples_json, labels_tsv)

    # A class to hold a labelled example in the test set.
    Example = namedtuple(
        "Example",
        [
            "extra_contexts",  # The previous contexts beyond the most recent.
            "context",  # The most recent context.
            "response",  # The response.
            "distractors",  # Incorrect responses used as distractors.
        ]
    )

    @property
    def examples(self):
        """The dataset, as a list of `TestReader.Example`s."""
        return self._examples

    def _read_data(self, examples_json, labels_tsv):
        """Read the data into memory, validating it."""
        with open(examples_json) as f:
            examples = json.loads(f.read())

        self._examples = []
        with open(labels_tsv) as f:
            tsv_reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for example, (
                    example_id, correct_candidate_id,
                    correct_candidate_text) in zip(examples, tsv_reader):
                example_id = int(example_id)
                assert example['example-id'] == example_id, (
                    "Expected equal example IDs: {} != {}").format(
                        example['example-id'], example_id)
                candidate_ids = {
                    candidate['candidate-id']: candidate['utterance']
                    for candidate in example['options-for-next']
                }
                assert correct_candidate_id in candidate_ids, (
                    "Correct candidate ID {} not in candidates for "
                    "example {}").format(correct_candidate_id, example_id)
                assert (
                    candidate_ids[correct_candidate_id]
                    == correct_candidate_text), (
                        "Correct text does not match {} != {}").format(
                            candidate_ids[correct_candidate_id],
                            correct_candidate_text)
                contexts = [
                    message['utterance']
                    for message in example['messages-so-far']
                ]
                distractors = [
                    candidate['utterance']
                    for candidate in example['options-for-next']
                    if candidate['candidate-id'] != correct_candidate_id
                ]
                self._examples.append(
                    self.Example(
                        extra_contexts=contexts[:-1],
                        context=contexts[-1],
                        response=correct_candidate_text,
                        distractors=distractors,
                    )
                )
