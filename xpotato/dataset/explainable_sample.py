from typing import Tuple, Dict
import networkx as nx
from xpotato.dataset.sample import Sample

from xpotato.graph_extractor.graph import PotatoGraph


class ExplainableSample(Sample):
    def __init__(
        self,
        example: Tuple[str, str],
        potato_graph: PotatoGraph = None,
        label_id: int = None,
    ) -> None:
        super().__init__(example=example, potato_graph=potato_graph, label_id=label_id)
        self.rationale = example[2]
        self.rationale_id = example[3]
        self.rationale_lemma = example[4]
        if len(example) >= 6:
            self.potato_graph = example[5]

    def _postprocess(self, graph: PotatoGraph) -> PotatoGraph:
        rationale_bool = []
        if len(self.rationale) != 0:
            for node, attr in graph.graph.nodes(data=True):
                if attr["name"] in self.rationale:
                    rationale_bool.append(True)
                else:
                    rationale_bool.append(False)
        nx.set_node_attributes(graph.graph, rationale_bool, "rationale")
        return graph

    def set_graph(self, graph: PotatoGraph) -> None:
        self.potato_graph = self._postprocess(graph)
