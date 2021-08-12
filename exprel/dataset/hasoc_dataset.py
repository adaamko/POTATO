import itertools
import logging
import pickle

import openpyxl
import pandas as pd
from exprel.database.db import Database
from exprel.dataset.dataset import Dataset
from exprel.dataset.hasoc_sample import HasocSample
from exprel.dataset.utils import amr_pn_to_graph
from sklearn import preprocessing
from tqdm import tqdm


class HasocDataset(Dataset):
    def __init__(self, path, lang="en", graph="amr"):
        super().__init__(lang)
        self.graph_format = graph
        if graph == "amr":
            import amrlib
            import spacy
            self.stog = amrlib.load_stog_model()
            # amrlib.setup_spacy_extension()
            self.nlp = spacy.load('en_core_web_md')
        self.le1 = preprocessing.LabelEncoder()
        self.le2 = preprocessing.LabelEncoder()
        self._dataset = [sample for sample in self.read_dataset(path)]

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = [sample for sample in self.read_dataset(value)]

    def set_graphs(self, graphs):
        for sample, graph in zip(self._dataset, graphs):
            sample.set_graph(graph)

    def read_dataset(self, df):
        for i, row in tqdm(enumerate(df.iterrows())):
            data = row[1]
            text = data.text
            task1 = data.task_1
            task2 = data.task_2
            ind = df.index[i]
            hasoc_sample = HasocSample(text, task1, task2, ind, self.nlp)
            yield hasoc_sample

    def to_dataframe(self):
        self.le1.fit([sample.task1 for sample in self._dataset])
        self.le2.fit([sample.task2 for sample in self._dataset])
        df = pd.DataFrame({"hasoc_id": [sample.hasoc_id for sample in self._dataset], "original_text": [
                          sample.original_text for sample in self._dataset],  "preprocessed_text": [sample.preprocessed_text for sample in self._dataset], "task1": [
            sample.task1 for sample in self._dataset], "task2": [sample.task2 for sample in self._dataset], "task1_id": self.le1.transform([sample.task1 for sample in self._dataset]), "task2_id": self.le2.transform([sample.task2 for sample in self._dataset]), "graph": [sample.graph for sample in self._dataset]})

        return df

    def one_versus_rest(self, df, entity):
        mapper = {entity: 1}

        one_versus_rest_df = df.copy()
        one_versus_rest_df["one_versus_rest"] = [
            mapper[item] if item in mapper else 0 for item in df.label]

        return one_versus_rest_df

    def parse_graphs(self, extractor, format="amr"):
        if format == "fourlang":
            graphs = list(extractor.parse_iterable(
                [sample.preprocessed_text for sample in self._dataset]))
            return graphs
        elif format == "amr":
            sens = [sample.preprocessed_text for sample in self._dataset]
            amr_graphs = []
            for sen in tqdm(sens):
                graphs = self.stog.parse_sents([sen])
                G, _ = amr_pn_to_graph(graphs[0], clean_nodes=True)
                amr_graphs.append(G)

            return amr_graphs

    def load_graphs(self, path):
        PIK = path

        with open(PIK, "rb") as f:
            self.graphs = pickle.load(f)

        self.set_graphs(self.graphs)

    def save_graphs(self, path):
        PIK = path
        with open(PIK, "wb") as f:
            pickle.dump(self.graphs, f)
