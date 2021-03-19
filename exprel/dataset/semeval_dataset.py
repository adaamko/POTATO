import itertools

import pandas as pd
from exprel.dataset.dataset import Dataset
from exprel.dataset.semeval_sample import SemevalSample
from sklearn import preprocessing
from tqdm import tqdm


class SemevalDataset(Dataset):
    def __init__(self, path, lang="en"):
        super().__init__(lang)
        self._dataset = [sample for sample in self.read_dataset(path)]
        self.le = preprocessing.LabelEncoder()

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = [sample for sample in self.read_dataset(value)]

    def read_dataset(self, path):
        with open(path, "r+") as f:
            for sample, label, _, _ in tqdm(itertools.zip_longest(*[f]*4)):
                sen_id, sentence = sample.split("\t")
                semeval_sample = SemevalSample(
                    sen_id, sentence.strip("\n"), label.strip("\n"), self.nlp)
                yield semeval_sample

    def to_dataframe(self):
        self.le.fit([sample.label for sample in self._dataset])
        df = pd.DataFrame({"sen_id": [sample.sen_id for sample in self._dataset], "e1": [sample.e1 for sample in self._dataset], "e2": [
                          sample.e2 for sample in self._dataset], "sentence": [sample.sentence for sample in self._dataset], "label": [sample.label for sample in self._dataset], "label_id": self.le.transform([sample.label for sample in self._dataset])})

        return df
