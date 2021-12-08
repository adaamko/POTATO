# Unsupervised mode

First you need to generate the training data, similar to the supervised mode

```python

from xpotato.dataset.dataset import Dataset

list_of_texts = ['Dogs are great', 'Cats are also great']
unlabeled = Dataset([(t, '') for t in list_of_texts], label_vocab={}, lang='en')
unlabeled.set_graphs(unlabeled.parse_graphs(graph_format='ud'))
df = unlabeled.to_dataframe()
df.to_pickle('dataset')
```

To run potato in unsupervised mode run:

```bash
streamlit run app.py -- -t train_dataset 
                        -v dev_dataset 
                        -g ud 
                        --mode unsupervised
```

And if you have rules already:

```bash
streamlit run app.py -- -t train_dataset 
                        -v dev_dataset 
                        -g ud 
                        -hr unsupervised_features.json 
                        --mode unsupervised
```

Once the frontend starts up and you define the labels, you are faced with the annotation interface. You can search elements by clicking on the appropriate column name and applying the desired filter. You can annotate instances by checking the checkbox at the beginning of the line. You can check multiple checkboxs at a time. Once you've selected the utterances you want to annotate, click on the _Annotate_ button. The annotated samples will appear in the lower table. You can clear the annotation of certain elements by selecting them in the second table and clicking _Clear annotation_.

Once you have some annotated data, you can train rules by clicking the _Train!_ button. It is recommended to set the _Rank features based on accuracy_ to True, if you have just a few samples. You will get a similar interface as in supervised mode, you can generate rule suggestions, and write your own rules as usual. Once you are satisfied with the rules, select each of them and click _annotate based on selected_. This process might take a while if you are working with large data. You should get all the rule matches marked in the first and the second tables. You can order the tables by each column, so it's easier to check. You will have to manually accept the annotations generated this way for them to appear in the second table.

