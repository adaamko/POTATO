from collections import defaultdict

import networkx as nx
import penman as pn
import re
from tuw_nlp.graph.utils import preprocess_node_alto


def amr_pn_to_graph(raw_dl, edge_attr='color', clean_nodes=False):
    g = pn.decode(raw_dl)
    G = nx.DiGraph()

    char_to_id = defaultdict(int)
    next_id = 0
    for i, trip in enumerate(g.triples):
        if i == 0:
            root_id = next_id
            if "-" in trip[2]:
                name = "-".join(trip[2].split("-")[:-1])
            else:
                name = trip[2]
            if clean_nodes:
                if name:
                    name = preprocess_node_alto(name)
                else:
                    name = "None"
            G.add_node(root_id, name=name)
            char_to_id[trip[0]] = next_id
            next_id += 1

        elif trip[1] == ":instance":
            if trip[2]:
                if "-" in trip[2]:
                    name = "-".join(trip[2].split("-")[:-1])
                else:
                    name = trip[2]
                if clean_nodes:
                    if name:
                        name = preprocess_node_alto(name)
                    else:
                        name = "None"
                G.add_node(next_id, name=name)
                char_to_id[trip[0]] = next_id
                next_id += 1

    for trip in g.triples:
        if trip[1] != ":instance":
            edge = "".join(trip[1].split(":")[1].split("-"))
            src = trip[0]
            tgt = trip[2]
            if src not in char_to_id:
                char_to_id[src] = next_id
                if "-" in src:
                    name = "-".join(src.split("-")[:-1])
                else:
                    name = src
                if clean_nodes:
                    if name:
                        name = preprocess_node_alto(name)
                    else:
                        name = "None"
                G.add_node(next_id, name=name)
                next_id += 1
            if tgt not in char_to_id:
                char_to_id[tgt] = next_id
                if "-" in tgt:
                    name = "-".join(tgt.split("-")[:-1])
                else:
                    name = tgt
                if clean_nodes:
                    if name:
                        name = preprocess_node_alto(name)
                    else:
                        name = "None"
                G.add_node(next_id, name=name)
                next_id += 1

            G.add_edge(char_to_id[src], char_to_id[tgt])
            G[char_to_id[src]][char_to_id[tgt]].update({edge_attr: edge})

    return G, root_id
