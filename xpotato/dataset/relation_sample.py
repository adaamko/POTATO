from typing import Tuple
import networkx as nx
from xpotato.dataset.sample import Sample


class RelationSample(Sample):
    def __init__(self, example: Tuple[str, str]) -> None:
        super().__init__(example=example)
        self.e1 = example[2]
        self.e2 = example[3]

    def _postprocess(self, graph: nx.Digraph) -> nx.Digraph:
        for node, attr in graph.nodes(data=True):
            if self.e1_lemma:
                if (
                    attr["name"] == self.e1_lemma
                    or attr["name"] == self.e1_lemma.split()[-1]
                ):
                    attr["name"] = "entity1"
            else:
                if attr["name"] == self.e1 or attr["name"] == self.e1.split()[-1]:
                    attr["name"] = "entity1"
            if self.e2_lemma:
                if (
                    attr["name"] == self.e2_lemma
                    or attr["name"] == self.e2_lemma.split()[-1]
                ):
                    attr["name"] = "entity2"
            else:
                if attr["name"] == self.e2 or attr["name"] == self.e2.split()[-1]:
                    attr["name"] = "entity2"

        return graph

    def set_graph(self, graph: nx.Digraph) -> None:
        self.graph = self._postprocess(graph)

    def prepare_lemma(self, doc) -> None:
        for token in doc.sentences[0].words:
            if token.text == self.e1:
                self.e1_lemma = token.lemma
            if token.text == self.e2:
                self.e2_lemma = token.lemma
