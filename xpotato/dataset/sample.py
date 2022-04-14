from typing import Dict, Tuple

from xpotato.graph_extractor.graph import PotatoGraph


class Sample:
    def __init__(
        self,
        example: Tuple[str, str],
        potato_graph: PotatoGraph = None,
        label_id: int = None,
    ) -> None:
        self.text = example[0]
        self.label = example[1]
        self.label_id = label_id
        self.potato_graph = potato_graph

    def set_graph(self, graph: PotatoGraph) -> None:
        self.potato_graph = graph

    def get_label_id(self, label_vocab: Dict[str, int]):
        if self.label_id is not None:
            return self.label_id
        elif self.label and self.label in label_vocab:
            return label_vocab[self.label]
        else:
            return None
