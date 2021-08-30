from typing import List, Tuple

import pandas as pd
import networkx as nx
import stanza
import pickle
from exprel.dataset.sample import Sample
from exprel.graph_extractor.extract import GraphExtractor


class Dataset:
    def __init__(self, examples: List[Tuple[str, str]], label_vocab: Dict[str, int], lang="en") -> None:
        self.nlp = stanza.Pipeline(lang)
        self.label_vocab = label_vocab
        self._dataset = self.read_dataset(examples)

    def read_dataset(self, examples: List[Tuple[str, str]]) -> List[Sample]:
        return [Sample(example) for example in examples]

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({"text": [sample.text for sample in self._dataset], "label": [
                          sample.label for sample in self._dataset], "label_id": [
                          self.label_vocab[sample.label] for sample in self._dataset] })
        return df

    def parse_graphs(self, extractor: GraphExtractor) -> List[nx.DiGraph]:
        graphs = list(extractor.parse_iterable(
            [sample.text for sample in self._dataset]))

        self.graphs = graphs
        return graphs

    def set_graphs(self, graphs: List[nx.DiGraph]) -> None:
        for sample, graph in zip(self._dataset, graphs):
            sample.set_graph(graph)

    def load_graphs(self, path: str) -> None:
        PIK = path

        with open(PIK, "rb") as f:
            self.graphs = pickle.load(f)

        self.set_graphs(self.graphs)

    def save_graphs(self, path: str) -> None:
        PIK = path
        with open(PIK, "wb") as f:
            pickle.dump(self.graphs, f)
