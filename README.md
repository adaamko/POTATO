<p align="center">
  <img src="https://raw.githubusercontent.com/adaamko/POTATO/dev/files/potato_logo.png" />
</p>

# POTATO: exPlainable infOrmation exTrAcTion framewOrk
POTATO is a human-in-the-loop XAI framework for extracting and evaluating interpretable graph features for any classification problem in Natural Language Processing.

## Built systems

To get started with rule-systems we provide rule-based features prebuilt with POTATO on different datasets (e.g. our paper _Offensive text detection on English Twitter with deep learning models and rule-based systems_ for the HASOC2021 shared task). If you are interested in that, you can go under _features/_ for more info!

## Install and Quick Start
### Setup
The tool is heavily dependent upon the [tuw-nlp](https://github.com/recski/tuw-nlp) repository. You can install tuw-nlp with pip:

```
pip install tuw-nlp
```
Then follow the [instructions](https://github.com/recski/tuw-nlp) to setup the package.


Then install POTATO from pip:

```
pip install xpotato
```

Or you can install it from source:

```
pip install -e .
```

### Usage

- POTATO is an IE tool that works on graphs, currently we support three types of graphs: AMR, UD and [Fourlang](https://github.com/kornai/4lang). 

- In the README we provide examples with fourlang semantic graphs. Make sure to follow the instructions in the [tuw_nlp](https://github.com/recski/tuw-nlp) repo to be able to build fourlang graphs. 

- If you are interested in AMR graphs, you can go to the [hasoc](https://github.com/adaamko/POTATO/tree/main/features/hasoc) folder To get started with rule-systems prebuilt with POTATO on the HASOC dataset (we also presented a paper named _Offensive text detection on English Twitter with deep learning models and rule-based systems_ for the HASOC2021 shared task). 

- We also provide experiments on the [CrowdTruth](https://github.com/CrowdTruth/Medical-Relation-Extraction) medical relation extraction datasets with UD graphs, go to the [crowdtruth](https://github.com/adaamko/POTATO/tree/main/features/crowdtruth) folder for more info!

- POTATO can also handle unlabeled, or partially labeled data, see [advanced](###advanced-mode) mode to get to know more.

__To see complete working examples go under the _notebooks/_ folder to see experiments on HASOC and on the Semeval relation extraction dataset.__

First import packages from potato:
```python
from xpotato.dataset.dataset import Dataset
from xpotato.models.trainer import GraphTrainer
```

First we demonstrate POTATO's capabilities with a few sentences manually picked from the dataset.

__Note that we replaced the two entitites in question with _XXX_ and _YYY_.__

```python
sentences = [("Governments and industries in nations around the world are pouring XXX into YYY.", "Entity-Destination(e1,e2)"),
            ("The scientists poured XXX into pint YYY.", "Entity-Destination(e1,e2)"),
            ("The suspect pushed the XXX into a deep YYY.", "Entity-Destination(e1,e2)"),
            ("The Nepalese government sets up a XXX to inquire into the alleged YYY of diplomatic passports.", "Other"),
            ("The entity1 to buy papers is pushed into the next entity2.", "Entity-Destination(e1,e2)"),
            ("An unnamed XXX was pushed into the YYY.", "Entity-Destination(e1,e2)"),
            ("Since then, numerous independent feature XXX have journeyed into YYY.", "Other"),
            ("For some reason, the XXX was blinded from his own YYY about the incommensurability of time.", "Other"),
            ("Sparky Anderson is making progress in his XXX from YYY and could return to managing the Detroit Tigers within a week.", "Other"),
            ("Olympics have already poured one XXX into the YYY.", "Entity-Destination(e1,e2)"),
            ("After wrapping him in a light blanket, they placed the XXX in the YYY his father had carved for him.", "Entity-Destination(e1,e2)"),
            ("I placed the XXX in a natural YYY, at the base of a part of the fallen arch.", "Entity-Destination(e1,e2)"),
            ("The XXX was delivered from the YYY of Lincoln Memorial on August 28, 1963 as part of his famous March on Washington.", "Other"),
            ("The XXX leaked from every conceivable YYY.", "Other"),
            ("The scientists placed the XXX in a tiny YYY which gets channelled into cancer cells, and is then unpacked with a laser impulse.", "Entity-Destination(e1,e2)"),
            ("The level surface closest to the MSS, known as the XXX, departs from an YYY by about 100 m in each direction.", "Other"),
            ("Gaza XXX recover from three YYY of war.", "Other"),
            ("This latest XXX from the animation YYY at Pixar is beautiful, masterly, inspired - and delivers a powerful ecological message.", "Other")]
```

Initialize the dataset and also provide a label encoding. Then parse the sentences into graphs. Currently we provide three types of graphs: _ud_, _fourlang_, _amr_. Also provide the language you want to parse, currently we support English (en) and German (de).

```python
dataset = Dataset(sentences, label_vocab={"Other":0, "Entity-Destination(e1,e2)": 1})
dataset.set_graphs(dataset.parse_graphs(graph_format="ud"), lang="en")
```

Check the dataset:
```python
df = dataset.to_dataframe()
```

We can also check any of the graphs:
### Check any of the graphs parsed

```python
from xpotato.models.utils import to_dot
from graphviz import Source

Source(to_dot(dataset.graphs[0]))
```
![graph](https://raw.githubusercontent.com/adaamko/POTATO/main/files/re_example.svg)

### Rules

If the dataset is prepared and the graphs are parsed, we can write rules to match labels. We can write rules either manually or extract
them automatically (POTATO also provides a frontend that tries to do both).

The simplest rule would be just a node in the graph:
```python
# The syntax of the rules is List[List[rules that we want to match], List[rules that shouldn't be in the matched graphs], Label of the rule]
rule_to_match = [[["(u_1 / into)"], [], "Entity-Destination(e1,e2)"]]
```

Init the rule matcher:
```python
from xpotato.graph_extractor.extract import FeatureEvaluator
evaluator = FeatureEvaluator()
```

Match the rules in the dataset:
```python
#match single feature
df = dataset.to_dataframe()
evaluator.match_features(df, rule_to_match)
```

|    | Sentence                                                                                                                        | Predicted label           | Matched rule                                        |
|---:|:--------------------------------------------------------------------------------------------------------------------------------|:--------------------------|:----------------------------------------------------|
|  0 | Governments and industries in nations around the world are pouring XXX into YYY.                                                | Entity-Destination(e1,e2) | [['(u_1 / into)'], [], 'Entity-Destination(e1,e2)'] |
|  1 | The scientists poured XXX into pint YYY.                                                                                        | Entity-Destination(e1,e2) | [['(u_1 / into)'], [], 'Entity-Destination(e1,e2)'] |
|  2 | The suspect pushed the XXX into a deep YYY.                                                                                     | Entity-Destination(e1,e2) | [['(u_1 / into)'], [], 'Entity-Destination(e1,e2)'] |
|  3 | The Nepalese government sets up a XXX to inquire into the alleged YYY of diplomatic passports.                                  | Entity-Destination(e1,e2) | [['(u_1 / into)'], [], 'Entity-Destination(e1,e2)'] |
|  4 | The entity1 to buy papers is pushed into the next entity2.                                                                      | Entity-Destination(e1,e2) | [['(u_1 / into)'], [], 'Entity-Destination(e1,e2)'] |
|  5 | An unnamed XXX was pushed into the YYY.                                                                                         | Entity-Destination(e1,e2) | [['(u_1 / into)'], [], 'Entity-Destination(e1,e2)'] |
|  6 | Since then, numerous independent feature XXX have journeyed into YYY.                                                           | Entity-Destination(e1,e2) | [['(u_1 / into)'], [], 'Entity-Destination(e1,e2)'] |
|  7 | For some reason, the XXX was blinded from his own YYY about the incommensurability of time.                                     |                           |                                                     |
|  8 | Sparky Anderson is making progress in his XXX from YYY and could return to managing the Detroit Tigers within a week.           |                           |                                                     |
|  9 | Olympics have already poured one XXX into the YYY.                                                                              | Entity-Destination(e1,e2) | [['(u_1 / into)'], [], 'Entity-Destination(e1,e2)'] |
| 10 | After wrapping him in a light blanket, they placed the XXX in the YYY his father had carved for him.                            |                           |                                                     |
| 11 | I placed the XXX in a natural YYY, at the base of a part of the fallen arch.                                                    |                           |                                                     |
| 12 | The XXX was delivered from the YYY of Lincoln Memorial on August 28, 1963 as part of his famous March on Washington.            |                           |                                                     |
| 13 | The XXX leaked from every conceivable YYY.                                                                                      |                           |                                                     |
| 14 | The scientists placed the XXX in a tiny YYY which gets channelled into cancer cells, and is then unpacked with a laser impulse. | Entity-Destination(e1,e2) | [['(u_1 / into)'], [], 'Entity-Destination(e1,e2)'] |
| 15 | The level surface closest to the MSS, known as the XXX, departs from an YYY by about 100 m in each direction.                   |                           |                                                     |
| 16 | Gaza XXX recover from three YYY of war.                                                                                         |                           |                                                     |
| 17 | This latest XXX from the animation YYY at Pixar is beautiful, masterly, inspired - and delivers a powerful ecological message.  |                           |                                                     |



You can see in the dataset that the rules only matched the instances where the "into" node was present.

One of the core features of our tool is that we are also able to match subgraphs. To describe a graph, we use the [PENMAN](https://github.com/goodmami/penman) notation. 

E.g. the string _(u_1 / into :1 (u_3 / pour))_ would describe a graph with two nodes ("into" and "pour") and a single directed edge with the label "1" between them.
```python
#match a simple graph feature
evaluator.match_features(df, [[["(u_1 / into :1 (u_2 / pour) :2 (u_3 / YYY))"], [], "Entity-Destination(e1,e2)"]])
```

Describing a subgraph with the string "(u_1 / into :1 (u_2 / pour) :2 (u_3 / YYY))" will return only three examples instead of 9 (when we only had a single node as a feature)
|    | Sentence                                                                                                                        | Predicted label           | Matched rule                                                                       |
|---:|:--------------------------------------------------------------------------------------------------------------------------------|:--------------------------|:-----------------------------------------------------------------------------------|
|  0 | Governments and industries in nations around the world are pouring XXX into YYY.                                                | Entity-Destination(e1,e2) | [['(u_1 / into :1 (u_2 / pour) :2 (u_3 / YYY))'], [], 'Entity-Destination(e1,e2)'] |
|  1 | The scientists poured XXX into pint YYY.                                                                                        | Entity-Destination(e1,e2) | [['(u_1 / into :1 (u_2 / pour) :2 (u_3 / YYY))'], [], 'Entity-Destination(e1,e2)'] |
|  2 | The suspect pushed the XXX into a deep YYY.                                                                                     |                           |                                                                                    |
|  3 | The Nepalese government sets up a XXX to inquire into the alleged YYY of diplomatic passports.                                  |                           |                                                                                    |
|  4 | The entity1 to buy papers is pushed into the next entity2.                                                                      |                           |                                                                                    |
|  5 | An unnamed XXX was pushed into the YYY.                                                                                         |                           |                                                                                    |
|  6 | Since then, numerous independent feature XXX have journeyed into YYY.                                                           |                           |                                                                                    |
|  7 | For some reason, the XXX was blinded from his own YYY about the incommensurability of time.                                     |                           |                                                                                    |
|  8 | Sparky Anderson is making progress in his XXX from YYY and could return to managing the Detroit Tigers within a week.           |                           |                                                                                    |
|  9 | Olympics have already poured one XXX into the YYY.                                                                              | Entity-Destination(e1,e2) | [['(u_1 / into :1 (u_2 / pour) :2 (u_3 / YYY))'], [], 'Entity-Destination(e1,e2)'] |
| 10 | After wrapping him in a light blanket, they placed the XXX in the YYY his father had carved for him.                            |                           |                                                                                    |
| 11 | I placed the XXX in a natural YYY, at the base of a part of the fallen arch.                                                    |                           |                                                                                    |
| 12 | The XXX was delivered from the YYY of Lincoln Memorial on August 28, 1963 as part of his famous March on Washington.            |                           |                                                                                    |
| 13 | The XXX leaked from every conceivable YYY.                                                                                      |                           |                                                                                    |
| 14 | The scientists placed the XXX in a tiny YYY which gets channelled into cancer cells, and is then unpacked with a laser impulse. |                           |                                                                                    |
| 15 | The level surface closest to the MSS, known as the XXX, departs from an YYY by about 100 m in each direction.                   |                           |                                                                                    |
| 16 | Gaza XXX recover from three YYY of war.                                                                                         |                           |                                                                                    |
| 17 | This latest XXX from the animation YYY at Pixar is beautiful, masterly, inspired - and delivers a powerful ecological message.  |                           |                                                                                    |


We can also add negated features that we don't want to match (e.g. this won't match the first row where 'pour' is present):
```python
#match a simple graph feature
evaluator.match_features(df, [[["(u_1 / into :2 (u_3 / YYY))"], ["(u_2 / pour)"], "Entity-Destination(e1,e2)"]])
```

|    | Sentence                                                                                                                        | Predicted label           | Matched rule                                                                     |
|---:|:--------------------------------------------------------------------------------------------------------------------------------|:--------------------------|:---------------------------------------------------------------------------------|
|  0 | Governments and industries in nations around the world are pouring XXX into YYY.                                                |                           |                                                                                  |
|  1 | The scientists poured XXX into pint YYY.                                                                                        |                           |                                                                                  |
|  2 | The suspect pushed the XXX into a deep YYY.                                                                                     | Entity-Destination(e1,e2) | [['(u_1 / into :2 (u_3 / YYY))'], ['(u_2 / pour)'], 'Entity-Destination(e1,e2)'] |
|  3 | The Nepalese government sets up a XXX to inquire into the alleged YYY of diplomatic passports.                                  | Entity-Destination(e1,e2) | [['(u_1 / into :2 (u_3 / YYY))'], ['(u_2 / pour)'], 'Entity-Destination(e1,e2)'] |
|  4 | The entity1 to buy papers is pushed into the next entity2.                                                                      |                           |                                                                                  |
|  5 | An unnamed XXX was pushed into the YYY.                                                                                         | Entity-Destination(e1,e2) | [['(u_1 / into :2 (u_3 / YYY))'], ['(u_2 / pour)'], 'Entity-Destination(e1,e2)'] |
|  6 | Since then, numerous independent feature XXX have journeyed into YYY.                                                           | Entity-Destination(e1,e2) | [['(u_1 / into :2 (u_3 / YYY))'], ['(u_2 / pour)'], 'Entity-Destination(e1,e2)'] |
|  7 | For some reason, the XXX was blinded from his own YYY about the incommensurability of time.                                     |                           |                                                                                  |
|  8 | Sparky Anderson is making progress in his XXX from YYY and could return to managing the Detroit Tigers within a week.           |                           |                                                                                  |
|  9 | Olympics have already poured one XXX into the YYY.                                                                              |                           |                                                                                  |
| 10 | After wrapping him in a light blanket, they placed the XXX in the YYY his father had carved for him.                            |                           |                                                                                  |
| 11 | I placed the XXX in a natural YYY, at the base of a part of the fallen arch.                                                    |                           |                                                                                  |
| 12 | The XXX was delivered from the YYY of Lincoln Memorial on August 28, 1963 as part of his famous March on Washington.            |                           |                                                                                  |
| 13 | The XXX leaked from every conceivable YYY.                                                                                      |                           |                                                                                  |
| 14 | The scientists placed the XXX in a tiny YYY which gets channelled into cancer cells, and is then unpacked with a laser impulse. |                           |                                                                                  |
| 15 | The level surface closest to the MSS, known as the XXX, departs from an YYY by about 100 m in each direction.                   |                           |                                                                                  |
| 16 | Gaza XXX recover from three YYY of war.                                                                                         |                           |                                                                                  |
| 17 | This latest XXX from the animation YYY at Pixar is beautiful, masterly, inspired - and delivers a powerful ecological message.  |                           |                                                                                  |

If we don't want to specify nodes, regex can also be used in place of the node and edge-names:

```python
#regex can be used to match any node (this will match instances where 'into' is connected to any node with '1' edge)
evaluator.match_features(df, [[["(u_1 / into :1 (u_2 / .*) :2 (u_3 / YYY))"], [], "Entity-Destination(e1,e2)"]])
```

|    | Sentence                                                                                                                        | Predicted label           | Matched rule                                                                     |
|---:|:--------------------------------------------------------------------------------------------------------------------------------|:--------------------------|:---------------------------------------------------------------------------------|
|  0 | Governments and industries in nations around the world are pouring XXX into YYY.                                                | Entity-Destination(e1,e2) | [['(u_1 / into :1 (u_2 / .*) :2 (u_3 / YYY))'], [], 'Entity-Destination(e1,e2)'] |
|  1 | The scientists poured XXX into pint YYY.                                                                                        | Entity-Destination(e1,e2) | [['(u_1 / into :1 (u_2 / .*) :2 (u_3 / YYY))'], [], 'Entity-Destination(e1,e2)'] |
|  2 | The suspect pushed the XXX into a deep YYY.                                                                                     | Entity-Destination(e1,e2) | [['(u_1 / into :1 (u_2 / .*) :2 (u_3 / YYY))'], [], 'Entity-Destination(e1,e2)'] |
|  3 | The Nepalese government sets up a XXX to inquire into the alleged YYY of diplomatic passports.                                  | Entity-Destination(e1,e2) | [['(u_1 / into :1 (u_2 / .*) :2 (u_3 / YYY))'], [], 'Entity-Destination(e1,e2)'] |
|  4 | The entity1 to buy papers is pushed into the next entity2.                                                                      |                           |                                                                                  |
|  5 | An unnamed XXX was pushed into the YYY.                                                                                         | Entity-Destination(e1,e2) | [['(u_1 / into :1 (u_2 / .*) :2 (u_3 / YYY))'], [], 'Entity-Destination(e1,e2)'] |
|  6 | Since then, numerous independent feature XXX have journeyed into YYY.                                                           | Entity-Destination(e1,e2) | [['(u_1 / into :1 (u_2 / .*) :2 (u_3 / YYY))'], [], 'Entity-Destination(e1,e2)'] |
|  7 | For some reason, the XXX was blinded from his own YYY about the incommensurability of time.                                     |                           |                                                                                  |
|  8 | Sparky Anderson is making progress in his XXX from YYY and could return to managing the Detroit Tigers within a week.           |                           |                                                                                  |
|  9 | Olympics have already poured one XXX into the YYY.                                                                              | Entity-Destination(e1,e2) | [['(u_1 / into :1 (u_2 / .*) :2 (u_3 / YYY))'], [], 'Entity-Destination(e1,e2)'] |
| 10 | After wrapping him in a light blanket, they placed the XXX in the YYY his father had carved for him.                            |                           |                                                                                  |
| 11 | I placed the XXX in a natural YYY, at the base of a part of the fallen arch.                                                    |                           |                                                                                  |
| 12 | The XXX was delivered from the YYY of Lincoln Memorial on August 28, 1963 as part of his famous March on Washington.            |                           |                                                                                  |
| 13 | The XXX leaked from every conceivable YYY.                                                                                      |                           |                                                                                  |
| 14 | The scientists placed the XXX in a tiny YYY which gets channelled into cancer cells, and is then unpacked with a laser impulse. |                           |                                                                                  |
| 15 | The level surface closest to the MSS, known as the XXX, departs from an YYY by about 100 m in each direction.                   |                           |                                                                                  |
| 16 | Gaza XXX recover from three YYY of war.                                                                                         |                           |                                                                                  |
| 17 | This latest XXX from the animation YYY at Pixar is beautiful, masterly, inspired - and delivers a powerful ecological message.  |                           |                                                                                  |

We can also train regex rules from a training data, this will automatically replace regex '.*' with nodes that are 
'good enough' statistically based on the provided dataframe.

```python
evaluator.train_feature("Entity-Destination(e1,e2)", "(u_1 / into :1 (u_2 / .*) :2 (u_3 / YYY))", df)
```

This returns '(u_1 / into :1 (u_2 / push|pour) :2 (u_3 / YYY))' (replaced '.*' with _push_ and _pour_)

### Learning rules

To extract rules automatically, train the dataset with graph features and rank them based on relevancy:

```python
df = dataset.to_dataframe()
trainer = GraphTrainer(df)
#extract features
features = trainer.prepare_and_train()

from sklearn.model_selection import train_test_split

train, val = train_test_split(df, test_size=0.2, random_state=1234)

#save train and validation, this is important for the frontend to work
train.to_pickle("train_dataset")
val.to_pickle("val_dataset")

import json

#also save the ranked features
with open("features.json", "w+") as f:
    json.dump(features, f)

```

You can also save the parsed graphs for evaluation or for caching:

```python
import pickle
with open("graphs.pickle", "wb") as f:
    pickle.dump(val.graph, f)
```

## Frontend

If the DataFrame is ready with the parsed graphs, the UI can be started to inspect the extracted rules and modify them. The frontend is a streamlit app, the simplest way of starting it is (the training and the validation dataset must be provided):

```
streamlit run frontend/app.py -- -t notebooks/train_dataset -v notebooks/val_dataset -g ud
```

it can be also started with the extracted features:

```
streamlit run frontend/app.py -- -t notebooks/train_dataset -v notebooks/val_dataset -g ud -sr notebooks/features.json
```

if you already used the UI and extracted the features manually and you want to load it, you can run:
```
streamlit run frontend/app.py -- -t notebooks/train_dataset -v notebooks/val_dataset -g ud -sr notebooks/features.json -hr notebooks/manual_features.json
```

### Advanced mode

If labels are not or just partially provided, the frontend can be started also in _advanced_ mode, where the user can _annotate_ a few examples at the start, then the system gradually offers rules based on the provided examples. 


Dataset without labels can be initialized with:
```python
sentences = [("Governments and industries in nations around the world are pouring XXX into YYY.", ""),
            ("The scientists poured XXX into pint YYY.", ""),
            ("The suspect pushed the XXX into a deep YYY.", ""),
            ("The Nepalese government sets up a XXX to inquire into the alleged YYY of diplomatic passports.", ""),
            ("The entity1 to buy papers is pushed into the next entity2.", ""),
            ("An unnamed XXX was pushed into the YYY.", ""),
            ("Since then, numerous independent feature XXX have journeyed into YYY.", ""),
            ("For some reason, the XXX was blinded from his own YYY about the incommensurability of time.", ""),
            ("Sparky Anderson is making progress in his XXX from YYY and could return to managing the Detroit Tigers within a week.", ""),
            ("Olympics have already poured one XXX into the YYY.", ""),
            ("After wrapping him in a light blanket, they placed the XXX in the YYY his father had carved for him.", ""),
            ("I placed the XXX in a natural YYY, at the base of a part of the fallen arch.", ""),
            ("The XXX was delivered from the YYY of Lincoln Memorial on August 28, 1963 as part of his famous March on Washington.", ""),
            ("The XXX leaked from every conceivable YYY.", ""),
            ("The scientists placed the XXX in a tiny YYY which gets channelled into cancer cells, and is then unpacked with a laser impulse.", ""),
            ("The level surface closest to the MSS, known as the XXX, departs from an YYY by about 100 m in each direction.", ""),
            ("Gaza XXX recover from three YYY of war.", ""),
            ("This latest XXX from the animation YYY at Pixar is beautiful, masterly, inspired - and delivers a powerful ecological message.", "")]
```


Then, the frontend can be started:
```
streamlit run frontend/app.py -- -t notebooks/unsupervised_dataset -g ud -m advanced
```

Once the frontend starts up and you define the labels, you are faced with the annotation interface. You can search elements by clicking on the appropriate column name and applying the desired filter. You can annotate instances by checking the checkbox at the beginning of the line. You can check multiple checkboxs at a time. Once you've selected the utterances you want to annotate, click on the _Annotate_ button. The annotated samples will appear in the lower table. You can clear the annotation of certain elements by selecting them in the second table and clicking _Clear annotation_.

Once you have some annotated data, you can train rules by clicking the _Train!_ button. It is recommended to set the _Rank features based on accuracy_ to True, if you have just a few samples. You will get a similar interface as in supervised mode, you can generate rule suggestions, and write your own rules as usual. Once you are satisfied with the rules, select each of them and click _annotate based on selected_. This process might take a while if you are working with large data. You should get all the rule matches marked in the first and the second tables. You can order the tables by each column, so it's easier to check. You will have to manually accept the annotations generated this way for them to appear in the second table.

- You can read about the use of the advanced mode in the [docs](https://github.com/adaamko/POTATO/tree/main/docs/README_advanced_mode.md)


## Evaluate
If you have the features ready and you want to evaluate them on a test set, you can run:

```python
python scripts/evaluate.py -t ud -f notebooks/features.json -d notebooks/val_dataset
```

The result will be a _csv_ file with the labels and the matched rules.

## Service
If you are ready with the extracted features and want to use our package in production for inference (generating predictions for sentences), we also provide a REST API built on POTATO (based on [fastapi](https://github.com/tiangolo/fastapi)).

First install FastAPI and [Uvicorn](https://www.uvicorn.org/)
```bash
pip install fastapi
pip install "uvicorn[standard]"
```

To start the service, you should set _language_, _graph\_type_ and the _features_  for the service. This can be done through enviroment variables.

Example:
```bash
export FEATURE_PATH=/home/adaamko/projects/POTATO/features/semeval/test_features.json
export GRAPH_FORMAT=ud
export LANG=en
```

Then, start the REST API:
```python
python services/main.py
```

It will start a service running on _localhost_ on port _8000_ (it will also initialize the correct models).

Then you can use any client to make post requests:
```bash
curl -X POST localhost:8000 -H 'Content-Type: application/json' -d '{"text":"The suspect pushed the XXX into a deep YYY.\nSparky Anderson is making progress in his XXX from YYY and could return to managing the Detroit Tigers within a week."}'
```

The answer will be a list with the predicted labels (if none of the rules match, it will return "NONE"):
```bash
["Entity-Destination(e1,e2)","NONE"]
```

The streamlit frontend also has an inference mode, where the implemented rule-system can be used for inference. It can be started with:

```bash
streamlit run frontend/app.py -- -hr features/semeval/test_features.json -m inference
```

## Contributing

We welcome all contributions! Please fork this repository and create a branch for your modifications. We suggest getting in touch with us first, by opening an issue or by writing an email to Adam Kovacs or Gabor Recski at firstname.lastname@tuwien.ac.at

## Citing

## License 

MIT license
