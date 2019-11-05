"""Tests for data.py.

Copyright PolyAI Limited.
"""

import unittest

from dstc7 import test_reader


class DataTest(unittest.TestCase):
    def test_test_reader(self):
        reader = test_reader.TestReader(
            examples_json="dstc7/testdata/test_examples.json",
            labels_tsv="dstc7/testdata/test_labels.tsv",
        )
        self.assertEqual(
            reader.examples,
            [
                test_reader.TestReader.Example(
                    extra_contexts=["0 context 1"],
                    context="0 context 2",
                    response="0 candidate A",
                    distractors=["0 candidate B", "0 candidate C"],
                ),
                test_reader.TestReader.Example(
                    extra_contexts=["1 context 1", "1 context 2"],
                    context="1 context 3",
                    response="1 candidate B",
                    distractors=["1 candidate A", "1 candidate C"],
                )
            ]
        )


if __name__ == "__main__":
    unittest.main()
