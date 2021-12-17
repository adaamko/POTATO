import pickle
from typing import List, Tuple, Dict

import networkx as nx
import pandas as pd

from xpotato.dataset.sample import Sample
from xpotato.graph_extractor.extract import GraphExtractor


class Dataset:
    def __init__(
        self, examples: List[Tuple[str, str]], label_vocab: Dict[str, int], lang="en"
    ) -> None:
        self.label_vocab = label_vocab
        self._dataset = self.read_dataset(examples)
        self.extractor = GraphExtractor(lang=lang, cache_fn=f"{lang}_nlp_cache")
        self.graphs = None

    def read_dataset(self, examples: List[Tuple[str, str]]) -> List[Sample]:
        return [Sample(example) for example in examples]

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "text": [sample.text for sample in self._dataset],
                "label": [sample.label for sample in self._dataset],
                "label_id": [
                    self.label_vocab[sample.label] if sample.label else None
                    for sample in self._dataset
                ],
                "graph": [sample.graph for sample in self._dataset],
            }
        )
        return df

    def parse_graphs(self, graph_format: str = "fourlang") -> List[nx.DiGraph]:
        graphs = list(
            self.extractor.parse_iterable(
                [sample.text for sample in self._dataset], graph_format
            )
        )

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
