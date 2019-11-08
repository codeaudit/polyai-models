# DSTC7 Ubuntu response ranking

This directory implements the DSTC7 ubuntu response ranking task, allowing
users to reproduce the results of the PolyAI contextual encoder on this task.

See also:
* https://github.com/IBM/dstc-noesis
* https://ibm.github.io/dstc-noesis/public/datasets.html


## Instructions

1. Download `Ubuntu_st1_test` and `Ubuntu_st1_ground_truth` from the
[dataset page](https://ibm.github.io/dstc-noesis/public/datasets.html).

You should now have a directory containing `ubuntu_responses_subtask_1.tsv` and
`ubuntu_test_subtask_1.json`.

2. Run the evaluation script

```bash
python dstc7/evaluate_encoder.py \
    --examples_json ubuntu_test_subtask_1.json \
    --labels_tsv ubuntu_responses_subtask_1.tsv \
    --encoder http://models.poly-ai.com/ubuntu_convert/v1/model.tar.gz
```

this should give the final output:

```
Recall@1 = 0.712
Recall@10 = 0.931
Recall@50 = 0.986
MRR = 0.788
```
