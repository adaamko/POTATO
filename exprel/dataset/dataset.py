import itertools
import stanza


class Dataset:
    def __init__(self, path, lang="en"):
        self.nlp = stanza.Pipeline(lang)

    def read_dataset(self, path):
        raise NotImplementedError("You need to specify a dataset type")

    def to_dataframe(self):
        raise NotImplementedError("You need to specify a dataset type")
