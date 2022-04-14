# Food-Disease dataset experiments

Dataset is from the [food-disease-dataset](https://github.com/gjorgjinac/food-disease-dataset.git) GitHub repository.

The __food_.ipynb__ jupyter notebook contains the preprocessing and the steps to build graphs for the dataset with the POTATO library.

Precomputed graphs can be downloaded running:

```bash
bash data.sh
```

Prebuilt rule-systems for both the _cause_ and the _treat_ label are also available in _food\_cause_features\_ud.json_ and _food\_treat\_features\_ud.json_.

Then the frontend of POTATO can be started from the __frontend__ directory:

```bash
streamlit run app.py -- -t ../features/food/food_train_dataset_cause_ud.tsv -v ../features/food/food_dev_dataset_cause_ud.tsv -hr ../features/crowdtruth/food_cause_features_ud.json
```

If you are done building the rule-system, you can evaluate it on the test data, for this run _evaluate.py_ from the _scripts_ directory.

```bash
python evaluate.py -t ud -f ../features/food/food_cause_features_ud.json -d ../features/crowdtruth/food_train_dataset_cause_ud.tsv
```