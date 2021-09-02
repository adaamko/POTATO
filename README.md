# EXPREL 
is a human-in-the-loop XAI framework for extracting and evaluating interpretable graph features for any classification problem  

## Install and Quick Start
### Setup
The tool is heavily dependent upon then [tuw-nlp](https://github.com/recski/tuw-nlp) repository:

```
git clone https://github.com/recski/tuw-nlp.git
cd tuw-nlp
pip install -e .
```

Then install EXPREL:
```
pip install -e .
```

### Usage

First import packages from exprel:
```python
from exprel.dataset.dataset import Dataset
from exprel.models.trainer import GraphTrainer
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
            ("all i want is for yara to survive tonight", "NOT")]
```

Initialize the dataset and also provide a label encoding. Then parse the sentences into graphs. Currently we provide three types of graphs: _ud_, _fourlang_, _amr_.

```python
dataset = Dataset(sentences, label_vocab={"NOT":0, "HOF": 1})
dataset.set_graphs(dataset.parse_graphs(graph_format="ud"))
```

Then, finally train the dataset with graph features and rank them based on relevancy:

```python
df = dataset.to_dataframe()
trainer = GraphTrainer(df)
#extract features
features = trainer.train()

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

## Evaluate
If you have the features ready and you want to evaluate them on a test set, you can run:

```python
python evaluate.py -t ud -f ../frontend/saved_features.json -d ../notebooks/val_dataset
```

The result will be a _csv_ file with the labels and the matched rules:
|   | Sentence                                | Predicted label | Matched rule                          |   |
|---|-----------------------------------------|-----------------|---------------------------------------|---|
| 0 | RT [USER]: ohhhh shit a [USER] [URL]    | HOF             | ['(u_48 / shit)']                     |   |
| 1 | [USER] you'd immediately stop           | HOF             | ['(u_40 / user :punct (u_42 / LSB))'] |   |
| 2 | fuck absolutely everything about today. | HOF             | Matched rule                          |   |

## Contributing

We welcome all contributions! Please fork this repository and create a branch for your modifications. We suggest getting in touch with us first, by opening an issue or by writing an email to Adam Kovacs or Gabor Recski at firstname.lastname@tuwien.ac.at

## Citing

## License 

MIT license
