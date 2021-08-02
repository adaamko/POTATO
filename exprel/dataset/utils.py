from collections import defaultdict

import networkx as nx
import penman as pn


def amr_pn_to_graph(raw_dl, edge_attr='color'):
    g = pn.decode(raw_dl)
    G = nx.DiGraph()

    char_to_id = defaultdict(int)
    next_id = 0
    for i, trip in enumerate(g.triples):
        if i == 0:
            root_id = next_id
            name = trip[2].split("-")[0]
            G.add_node(root_id, name=name)
            char_to_id[trip[0]] = next_id
            next_id += 1

        elif trip[1] == ":instance":
            G.add_node(next_id, name=trip[2].split("-")[0])
            char_to_id[trip[0]] = next_id
            next_id += 1

    for trip in g.triples:
        if trip[1] != ":instance":
            edge = trip[1].split(":")[1]
            src = trip[0]
            tgt = trip[2]
            if src not in char_to_id:
                char_to_id[src] = next_id
                G.add_node(next_id, name=src.split("-")[0])
                next_id += 1
            if tgt not in char_to_id:
                char_to_id[tgt] = next_id
                G.add_node(next_id, name=tgt.split("-")[0])
                next_id += 1

            G.add_edge(char_to_id[src], char_to_id[tgt])
            G[char_to_id[src]][char_to_id[tgt]].update({edge_attr: edge})

    return G, root_id
