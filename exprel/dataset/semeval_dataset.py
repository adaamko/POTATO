import itertools
import pickle

import pandas as pd
from exprel.dataset.dataset import Dataset
from exprel.dataset.semeval_sample import SemevalSample
from sklearn import preprocessing
from tqdm import tqdm


class SemevalDataset(Dataset):
    def __init__(self, path, lang="en"):
        super().__init__(lang)
        self.le = preprocessing.LabelEncoder()
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

    def read_dataset(self, path):
        with open(path, "r+") as f:
            for sample, label, _, _ in tqdm(itertools.zip_longest(*[f]*4)):
                sen_id, sentence = sample.split("\t")
                semeval_sample = SemevalSample(
                    sen_id, sentence.strip("\n"), label.strip("\n"), self.nlp, self.docs)
                yield semeval_sample

    def to_dataframe(self):
        self.le.fit([sample.label for sample in self._dataset])
        df = pd.DataFrame({"sen_id": [sample.sen_id for sample in self._dataset], "e1": [sample.e1 for sample in self._dataset], "e2": [
                          sample.e2 for sample in self._dataset],  "e1_lemma": [sample.e1_lemma for sample in self._dataset], "e2_lemma": [
            sample.e2_lemma for sample in self._dataset], "sentence": [sample.sentence for sample in self._dataset], "label": [sample.label for sample in self._dataset], "label_id": self.le.transform([sample.label for sample in self._dataset]), "graph": [sample.graph for sample in self._dataset]})

        return df

    def one_versus_rest(self, df, entity):
        mapper = {entity: 1}

        one_versus_rest_df = df.copy()
        one_versus_rest_df["one_versus_rest"] = [
            mapper[item] if item in mapper else 0 for item in df.label]

        return one_versus_rest_df

    def parse_graphs(self, extractor):
        graphs = list(extractor.parse_iterable([sample.sentence for sample in self._dataset]))
        return graphs

    def load_graphs(self, path):
        PIK = path

        with open(PIK, "rb") as f:
            self.graphs = pickle.load(f)

        self.set_graphs(self.graphs)

    def save_graphs(self, path):
        PIK = path
        with open(PIK, "wb") as f:
            pickle.dump(self.graphs, f)
