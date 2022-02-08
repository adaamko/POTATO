from typing import Tuple

from xpotato.graph_extractor.graph import PotatoGraph


class Sample:
    def __init__(
        self, example: Tuple[str, str], potato_graph: PotatoGraph = None
    ) -> None:
        self.text = example[0]
        self.label = example[1]
        self.potato_graph = potato_graph

    def set_graph(self, graph: PotatoGraph) -> None:
        self.potato_graph = graph
