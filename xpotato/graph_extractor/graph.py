from tuw_nlp.graph.graph import Graph
from tuw_nlp.graph.utils import check_if_str_is_penman
import networkx as nx
import json
from typing import Dict, Union


class PotatoGraph:
    def __init__(self, graph: Union[str, Dict, Graph, nx.DiGraph] = None) -> None:
        if type(graph) == str and check_if_str_is_penman(graph):
            self.graph = Graph.from_penman(graph)
        elif type(graph) == str and not check_if_str_is_penman(graph):
            json_graph = json.loads(graph)
            self.graph = Graph.from_json(json_graph)
        elif type(graph) == dict:
            self.graph = Graph.from_json(graph)
        elif type(graph) == nx.DiGraph:
            self.graph = Graph.from_networkx(graph)
        elif Graph in type(graph).__bases__:
            self.graph = graph
        elif graph is None:
            self.graph = None
        else:
            raise Exception("Unknown graph type")

    def __str__(self) -> str:
        if self.graph is None:
            return "Empty graph"
        else:
            return self.graph.to_penman()

    def to_dict(self) -> dict:
        return self.graph.to_json()

    def prune(self) -> None:
        self.graph.prune_graphs()
