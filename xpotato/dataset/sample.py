from typing import Tuple

import networkx as nx


class Sample:
    def __init__(self, example: Tuple[str, str]) -> None:
        self.text = example[0]
        self.label = example[1]
        self.graph = None

    def set_graph(self, graph: nx.DiGraph) -> None:
        self.graph = graph
