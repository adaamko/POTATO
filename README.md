# POTATO: exPlainable infOrmation exTrAcTion framewOrk
POTATO is a human-in-the-loop XAI framework for extracting and evaluating interpretable graph features for any classification problem  

## Built systems

To get started with rule-systems we provide rule-based features prebuilt with POTATO on different datasets (e.g. our paper _Offensive text detection on English Twitter with deep learning models and rule-based systems_ for the HASOC2021 shared task). If you are interested in that, you can go under _features/_ for more info!

## Install and Quick Start
### Setup
The tool is heavily dependent upon the [tuw-nlp](https://github.com/recski/tuw-nlp) repository:

```
git clone https://github.com/recski/tuw-nlp.git
cd tuw-nlp
pip install -e .
```

Then install POTATO:
```
pip install -e .
```

### Usage

First import packages from potato:
```python
from potato.dataset.dataset import Dataset
from potato.models.trainer import GraphTrainer
```

Initialize the dataset you want to classify:
```python
sentences = [("fuck absolutely everything about today.", "HOF"),
            ("I just made food and I'm making myself sick to my stomach. Lol, wtf is this shit", "HOF"),
            ("RT [USER]: America is the most fucked up country [URL]", "HOF"),
            ("you'd be blind to not see the heart eyes i have for you.", "NOT"),
            ("It's hard for me to give a fuck now", "HOF"),
            ("tell me everything", "NOT"),
            ("Bitch YES [URL]", "HOF"),
            ("Eight people a minute....", "NOT"),
            ("RT [USER]: im not fine, i need you", "NOT"),
            ("Holy shit.. 3 months and I'll be in Italy", "HOF"),
            ("Now I do what I want 🤪", "NOT"),
            ("[USER] you'd immediately stop", "NOT"),
            ("Just... shut the fuck up", "HOF"),
            ("RT [USER]: ohhhh shit a [USER] [URL]", "HOF"),
            ("all i want is for yara to survive tonight", "NOT"),
            ("fuck them", "HOF")]
```

Initialize the dataset and also provide a label encoding. Then parse the sentences into graphs. Currently we provide three types of graphs: _ud_, _fourlang_, _amr_.

```python
dataset = Dataset(sentences, label_vocab={"NOT":0, "HOF": 1})
dataset.set_graphs(dataset.parse_graphs(graph_format="ud"))
```

### Rules

If the dataset is prepared and the graphs are parsed, we can write rules to match labels. We can write rules either manually or extract
them automatically (POTATO also provides a frontend that tries to do both).

The simplest rule would be just a node in the graph:
```python
#the syntax of the rules is List[List[rules that we want to match], List[rules that shouldn't be in the matched graphs], Label of the rule]
rule_to_match = [[["(u_1 / fuck)"], [], "HOF"]]
```

Init the rule matcher:
```python
from potato.graph_extractor.extract import FeatureEvaluator
evaluator = FeatureEvaluator()
```

Match the rules in the dataset:
```python
#match single feature
df = dataset.to_dataframe()
evaluator.match_features(df, rule_to_match)
```

The function will return a dataframe with the matched instances:
|    | Sentence                                                                         | Predicted label   | Matched rule                  |
|---:|:---------------------------------------------------------------------------------|:------------------|:------------------------------|
|  0 | fuck absolutely everything about today.                                          | HOF               | [['(u_1 / fuck)'], [], 'HOF'] |
|  1 | I just made food and I'm making myself sick to my stomach. Lol, wtf is this shit |                   |                               |
|  2 | RT [USER]: America is the most fucked up country [URL]                           |                   |                               |
|  3 | you'd be blind to not see the heart eyes i have for you.                         |                   |                               |
|  4 | It's hard for me to give a fuck now                                              | HOF               | [['(u_1 / fuck)'], [], 'HOF'] |
|  5 | tell me everything                                                               |                   |                               |
|  6 | Bitch YES [URL]                                                                  |                   |                               |
|  7 | Eight people a minute....                                                        |                   |                               |
|  8 | RT [USER]: im not fine, i need you                                               |                   |                               |
|  9 | Holy shit.. 3 months and I'll be in Italy                                        |                   |                               |
| 10 | Now I do what I want 🤪                                                          |                   |                               |
| 11 | [USER] you'd immediately stop                                                    |                   |                               |
| 12 | Just... shut the fuck up                                                         | HOF               | [['(u_1 / fuck)'], [], 'HOF'] |
| 13 | RT [USER]: ohhhh shit a [USER] [URL]                                             |                   |                               |
| 14 | all i want is for yara to survive tonight                                        |                   |                               |
| 15 | fuck them                                                                        | HOF               | [['(u_1 / fuck)'], [], 'HOF'] |

On of the core features of our tool is that we are also able to match subgraphs:
```python
#match a simple graph feature
evaluator.match_features(df, [[["(u_1 / fuck :obj (u_2 / everything))"], [], "HOF"]])
```

This will only return one match instead of three:
|    | Sentence                                                                         | Predicted label   | Matched rule                                          |
|---:|:---------------------------------------------------------------------------------|:------------------|:------------------------------------------------------|
|  0 | fuck absolutely everything about today.                                          | HOF               | [['(u_1 / fuck :obj (u_2 / everything))'], [], 'HOF'] |
|  1 | I just made food and I'm making myself sick to my stomach. Lol, wtf is this shit |                   |                                                       |
|  2 | RT [USER]: America is the most fucked up country [URL]                           |                   |                                                       |
|  3 | you'd be blind to not see the heart eyes i have for you.                         |                   |                                                       |
|  4 | It's hard for me to give a fuck now                                              |                   |                                                       |
|  5 | tell me everything                                                               |                   |                                                       |
|  6 | Bitch YES [URL]                                                                  |                   |                                                       |
|  7 | Eight people a minute....                                                        |                   |                                                       |
|  8 | RT [USER]: im not fine, i need you                                               |                   |                                                       |
|  9 | Holy shit.. 3 months and I'll be in Italy                                        |                   |                                                       |
| 10 | Now I do what I want 🤪                                                          |                   |                                                       |
| 11 | [USER] you'd immediately stop                                                    |                   |                                                       |
| 12 | Just... shut the fuck up                                                         |                   |                                                       |
| 13 | RT [USER]: ohhhh shit a [USER] [URL]                                             |                   |                                                       |
| 14 | all i want is for yara to survive tonight                                        |                   |                                                       |
| 15 | fuck them                                                                        |                   |                                                       |                                                                   | HOF               | [['(u_1 / fuck)'], ['(u_2 / absolutely)'], 'HOF'] |


We can also add negated features that we don't want to match (this won't match the first row where 'absolutely' is present):
```python
#match a simple graph feature
evaluator.match_features(df, [[["(u_1 / fuck)"], ["(u_2 / absolutely)"], "HOF"]])
```

|    | Sentence                                                                         | Predicted label   | Matched rule                                      |
|---:|:---------------------------------------------------------------------------------|:------------------|:--------------------------------------------------|
|  0 | fuck absolutely everything about today.                                          |                   |                                                   |
|  1 | I just made food and I'm making myself sick to my stomach. Lol, wtf is this shit |                   |                                                   |
|  2 | RT [USER]: America is the most fucked up country [URL]                           |                   |                                                   |
|  3 | you'd be blind to not see the heart eyes i have for you.                         |                   |                                                   |
|  4 | It's hard for me to give a fuck now                                              | HOF               | [['(u_1 / fuck)'], ['(u_2 / absolutely)'], 'HOF'] |
|  5 | tell me everything                                                               |                   |                                                   |
|  6 | Bitch YES [URL]                                                                  |                   |                                                   |
|  7 | Eight people a minute....                                                        |                   |                                                   |
|  8 | RT [USER]: im not fine, i need you                                               |                   |                                                   |
|  9 | Holy shit.. 3 months and I'll be in Italy                                        |                   |                                                   |
| 10 | Now I do what I want 🤪                                                          |                   |                                                   |
| 11 | [USER] you'd immediately stop                                                    |                   |                                                   |
| 12 | Just... shut the fuck up                                                         | HOF               | [['(u_1 / fuck)'], ['(u_2 / absolutely)'], 'HOF'] |
| 13 | RT [USER]: ohhhh shit a [USER] [URL]                                             |                   |                                                   |
| 14 | all i want is for yara to survive tonight                                        |                   |                                                   |
| 15 | fuck them                                                                        | HOF               | [['(u_1 / fuck)'], ['(u_2 / absolutely)'], 'HOF'] |

If we don't want to specify nodes, regex can also be used in place of the node and edge-names:

```python
#regex can be used to match any node (this will match instances where 'fuck' is connected to any node with 'obj' edge)
evaluator.match_features(df, [[["(u_1 / fuck :obj (u_2 / .*))"], [], "HOF"]])
```

|    | Sentence                                                                         | Predicted label   | Matched rule                                  |
|---:|:---------------------------------------------------------------------------------|:------------------|:----------------------------------------------|
|  0 | fuck absolutely everything about today.                                          | HOF               | [['(u_1 / fuck :obj (u_2 / .*))'], [], 'HOF'] |
|  1 | I just made food and I'm making myself sick to my stomach. Lol, wtf is this shit |                   |                                               |
|  2 | RT [USER]: America is the most fucked up country [URL]                           |                   |                                               |
|  3 | you'd be blind to not see the heart eyes i have for you.                         |                   |                                               |
|  4 | It's hard for me to give a fuck now                                              |                   |                                               |
|  5 | tell me everything                                                               |                   |                                               |
|  6 | Bitch YES [URL]                                                                  |                   |                                               |
|  7 | Eight people a minute....                                                        |                   |                                               |
|  8 | RT [USER]: im not fine, i need you                                               |                   |                                               |
|  9 | Holy shit.. 3 months and I'll be in Italy                                        |                   |                                               |
| 10 | Now I do what I want 🤪                                                          |                   |                                               |
| 11 | [USER] you'd immediately stop                                                    |                   |                                               |
| 12 | Just... shut the fuck up                                                         |                   |                                               |
| 13 | RT [USER]: ohhhh shit a [USER] [URL]                                             |                   |                                               |
| 14 | all i want is for yara to survive tonight                                        |                   |                                               |
| 15 | fuck them                                                                        | HOF               | [['(u_1 / fuck :obj (u_2 / .*))'], [], 'HOF'] |

We can also train regex rules from a training data, this will automatically replace regex '.*' with nodes that are 
'good enough' statistically based on the provided dataframe.

```python
#regex can be used to match any node (this will match instances where 'fuck' is connected to any node with 'obj' edge)
evaluator.train_feature("HOF", "(u_1 / fuck :obj (u_2 / .*))", df)
```

This will return '(u_1 / fuck :obj (u_2 / everything|they))'] (replaced '.*' with _everything_ and _they_)

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

To see the code you can check the jupyter notebook under *notebooks/examples.ipynb*

## Frontend

If the DataFrame is ready with the parsed graphs, the UI can be started to inspect the extracted rules and modify them. The frontend is a streamlit app, the simplest way of starting it is (the training and the validation dataset must be provided):

```
streamlit run app.py -- -t ../notebooks/train_dataset -v ../notebooks/val_dataset -g ud
```

it can be also started with the extracted features:

```
streamlit run app.py -- -t ../notebooks/train_dataset -v ../notebooks/val_dataset -g ud -sr ../notebooks/features.json
```

if you already used the UI and extracted the features manually and you want to load it, you can run:
```
streamlit run app.py -- -t ../notebooks/train_dataset -v ../notebooks/val_dataset -g ud -sr ../notebooks/features.json -hr ../notebooks/manual_features.json
```

### Unsupervised mode

If labels are not or just partially provided, the frontend can be started also in _unsupervised_ mode, where the user can _annotate_ a few examples at the start, then the system gradually offers rules based on the provided examples. 


Dataset without labels can be initialized with:
```python
sentences = [("fuck absolutely everything about today.", ""),
            ("I just made food and I'm making myself sick to my stomach. Lol, wtf is this shit", ""),
            ("RT [USER]: America is the most fucked up country [URL]", ""),
            ("you'd be blind to not see the heart eyes i have for you.", ""),
            ("It's hard for me to give a fuck now", ""),
            ("tell me everything", ""),
            ("Bitch YES [URL]", ""),
            ("Eight people a minute....", ""),
            ("RT [USER]: im not fine, i need you", ""),
            ("Holy shit.. 3 months and I'll be in Italy", ""),
            ("Now I do what I want 🤪", ""),
            ("[USER] you'd immediately stop", ""),
            ("Just... shut the fuck up", ""),
            ("RT [USER]: ohhhh shit a [USER] [URL]", ""),
            ("all i want is for yara to survive tonight", ""),
            ("fuck them", "")]
```

Then, the frontend can be started:
```
streamlit run app.py -- -t ../notebooks/test_dataset.pickle -g ud -m unsupervised
```



## Evaluate
If you have the features ready and you want to evaluate them on a test set, you can run:

```python
python evaluate.py -t ud -f ../frontend/saved_features.json -d ../notebooks/val_dataset
```

The result will be a _csv_ file with the labels and the matched rules:
|     | Sentence                                | Predicted label | Matched rule                          |     |
| --- | --------------------------------------- | --------------- | ------------------------------------- | --- |
| 0   | RT [USER]: ohhhh shit a [USER] [URL]    | HOF             | ['(u_48 / shit)']                     |     |
| 1   | [USER] you'd immediately stop           | HOF             | ['(u_40 / user :punct (u_42 / LSB))'] |     |
| 2   | fuck absolutely everything about today. | HOF             |  ['(u_1 / fuck)']                            |     |

## Contributing

We welcome all contributions! Please fork this repository and create a branch for your modifications. We suggest getting in touch with us first, by opening an issue or by writing an email to Adam Kovacs or Gabor Recski at firstname.lastname@tuwien.ac.at

## Citing

## License 

MIT license
