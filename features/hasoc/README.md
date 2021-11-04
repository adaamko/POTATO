# HASOC experiments

We also participated in the HASOC2021: Hate Speech and Offensive Content Identification in English and Indo-Aryan Languages shared task. You can download the dataset from the official [site](https://hasocfire.github.io/hasoc/2021/index.html).

Both the notebook for preprocessing the data and our output for the task can be found under the _hasoc2021_ branch.

Precomputed graphs can be downloaded running:

```bash
bash data.sh
```

Prebuilt rule-systems are available in this directory for the _2019, 2020, 2021_ tasks. The features are mainly only built for the binary _HOF/NOT_ labels.

Then the frontend of POTATO can be started from the __frontend__ directory:

```bash
streamlit run app.py -- -t ../features/hasoc/hasoc_2021_train_amr.pickle -v ../features/hasoc/hasoc_2021_val_amr.pickle -hr ../features/hasoc/2021_train_features_task1.json
```

If you want to reproduce our output run _evaluate.py_ from the _scripts_ directory.

```bash
python evaluate.py -t ud -f ../features/hasoc/2021_train_features_task1.json -d ../features/food/hasoc_2021_test_amr.pickle
```
