from re import I
from typing import List, Tuple, Dict

import networkx as nx
import pandas as pd

from tqdm import tqdm
from tuw_nlp.graph.utils import graph_to_pn
from xpotato.dataset.sample import Sample
from xpotato.graph_extractor.extract import GraphExtractor
from xpotato.graph_extractor.graph import PotatoGraph


class Dataset:
    def __init__(
        self,
        examples: List[Tuple[str, str]] = None,
        label_vocab: Dict[str, int] = {},
        lang="en",
        path=None,
        binary=False,
    ) -> None:
        self.label_vocab = label_vocab
        if path:
            self._dataset = self.read_dataset(path=path, binary=binary)
        else:
            self._dataset = self.read_dataset(examples=examples)
        self.extractor = GraphExtractor(lang=lang, cache_fn=f"{lang}_nlp_cache")
        self.graphs = None

    def read_dataset(
        self,
        examples: List[Tuple[str, str]] = None,
        path: str = None,
        binary: bool = False,
    ) -> List[Sample]:
        if examples:
            return [Sample(example) for example in examples]
        elif path:
            if binary:
                df = pd.read_pickle(path)
                graphs = []
                for i, graph in enumerate(df["graph"].tolist()):
                    graph.remove_nodes_from(list(nx.isolates(graph)))
                    # ADAM: THIS IS JUST FOR PICKLE TO PENMAN CONVERSION
                    graph = self._random_postprocess(graph)

                    g = [
                        c
                        for c in sorted(
                            nx.weakly_connected_components(graph), key=len, reverse=True
                        )
                    ]
                    if len(g) > 1:
                        print(
                            "WARNING: graph has multiple connected components, taking the largest"
                        )
                        g_pn = graph_to_pn(graph.subgraph(g[0].copy()))
                    else:
                        g_pn = graph_to_pn(graph)

                    graphs.append(g_pn)
                df.drop(columns=["graph"], inplace=True)
                df["graph"] = graphs
            else:
                df = pd.read_csv(path, sep="\t")

            return [
                Sample(
                    (example["text"], example["label"]),
                    PotatoGraph(graph_str=example["graph"]),
                )
                for _, example in tqdm(df.iterrows())
            ]
        else:
            raise ValueError("No examples or path provided")

    # ADAM: THIS WILL NEED TO BE ADDRESSED
    def _random_postprocess(self, graph: nx.DiGraph) -> nx.DiGraph:
        for node, attr in graph.nodes(data=True):
            if len(attr["name"].split()) > 1:
                attr["name"] = attr["name"].split()[0]

        return graph

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "text": [sample.text for sample in self._dataset],
                "label": [sample.label for sample in self._dataset],
                "label_id": [
                    self.label_vocab[sample.label] if sample.label else None
                    for sample in self._dataset
                ],
                "graph": [sample.potato_graph.graph for sample in self._dataset],
            }
        )
        return df

    def parse_graphs(self, graph_format: str = "fourlang") -> List[nx.DiGraph]:
        graphs = list(
            self.extractor.parse_iterable(
                [sample.text for sample in self._dataset], graph_format
            )
        )

        self.graphs = [PotatoGraph(graph) for graph in graphs]
        return self.graphs

    def set_graphs(self, graphs: List[PotatoGraph]) -> None:
        for sample, potato_graph in zip(self._dataset, graphs):
            potato_graph.graph.remove_edges_from(nx.selfloop_edges(potato_graph.graph))
            sample.set_graph(potato_graph)

    def load_graphs(self, path: str) -> None:
        with open(path, "rb") as f:
            for line in f:
                graph = PotatoGraph()
                graph.from_penman(line.strip())
                self.graphs.append(graph)

        self.set_graphs(self.graphs)

    def save_dataset(self, path: str) -> None:
        df = pd.DataFrame(
            {
                "text": [sample.text for sample in self._dataset],
                "label": [sample.label for sample in self._dataset],
                "label_id": [
                    self.label_vocab[sample.label] if sample.label else None
                    for sample in self._dataset
                ],
                "graph": [str(sample.potato_graph) for sample in self._dataset],
            }
        )
        df.to_csv(path, index=False, sep="\t")

    def save_graphs(self, path: str) -> None:
        with open(path, "wb") as f:
            for graph in self.graphs:
                f.write(str(graph) + "\n")
