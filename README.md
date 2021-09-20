# EXPREL
## This branch contains the code for the rule system submitted to HASOC2021 shared task in the state of the submission

For detailed instructions, go to the _main_ branch.

The _notebooks/hasoc2020.ipnyb_ contains our experiments. If you want to evaluate a given ruleset, run the following command:

```bash
python evaluate.py -t amr -f /home/kovacs/projects/exp-relation-extraction/notebooks/features/2021_train_features_task1.json -d /home/kovacs/projects/exp-relation-extraction/data/hasoc_2021_test_normalized.csv -g /home/kovacs/projects/exp-relation-extraction/notebooks/graphs/hasoc2021_test_amr.pickle > 2021_test_predicted.tsv
```
