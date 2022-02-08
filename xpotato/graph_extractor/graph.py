import networkx as nx
from tuw_nlp.graph.utils import graph_to_pn
from xpotato.dataset.utils import default_pn_to_graph


class PotatoGraph:
    def __init__(self, graph: nx.DiGraph = None, graph_str: str = None) -> None:
        if graph:
            self.graph = graph
        elif graph_str:
            self.graph, _ = default_pn_to_graph(graph_str)
        else:
            self.graph = None

    def __str__(self) -> str:
        return graph_to_pn(self.graph)
