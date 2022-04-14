# CrowdTruth data experiments

Dataset is from the [CrowdTruth](https://github.com/CrowdTruth/Medical-Relation-Extraction) GitHub repository.

The __crowdtruth.ipynb__ jupyter notebook contains the preprocessing and the steps to build graphs for the dataset with the POTATO library.

Precomputed graphs can be downloaded running:

```bash
bash data.sh
```

Prebuilt rule-systems for both the _cause_ and the _treat_ label are also available in _crowd\_cause_features\_ud.json_ and _crowd\_treat\_features\_ud.json_.

Then the frontend of POTATO can be started from the __frontend__ directory:

```bash
streamlit run app.py -- -t ../features/crowdtruth/crowdtruth_train_dataset_cause_ud.tsv -v ../features/crowdtruth/crowdtruth_dev_dataset_cause_ud.tsv -hr ../features/crowdtruth/crowd_cause_features_ud.json
```

If you are done building the rule-system, you can evaluate it on the test data, for this run _evaluate.py_ from the _scripts_ directory.

```bash
python evaluate.py -t ud -f ../features/crowdtruth/crowd_cause_features_ud.json -d ../features/crowdtruth/crowdtruth_train_dataset_cause_ud.tsv
```