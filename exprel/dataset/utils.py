from collections import defaultdict

import networkx as nx
import penman as pn
import re


def d_clean(string):
    s = string
    for c in '\\=@-,\'".!:;<>/{}[]()#^?':
        s = s.replace(c, '_')
    s = s.replace('$', '_dollars')
    s = s.replace('%', '_percent')
    s = s.replace('|', ' ')
    s = s.replace('*', ' ')
    if s == '#':
        s = '_number'
    keywords = ("graph", "node", "strict", "edge")
    if re.match('^[0-9]', s) or s in keywords:
        s = "X" + s

    if not s:
        return "None"
    return s


def amr_pn_to_graph(raw_dl, edge_attr='color'):
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
            G.add_node(root_id, name=d_clean(name))
            char_to_id[trip[0]] = next_id
            next_id += 1

        elif trip[1] == ":instance":
            if trip[2]:
                if "-" in trip[2]:
                    name = "-".join(trip[2].split("-")[:-1])
                else:
                    name = trip[2]
                G.add_node(next_id, name=d_clean(name))
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
                G.add_node(next_id, name=d_clean(name))
                next_id += 1
            if tgt not in char_to_id:
                char_to_id[tgt] = next_id
                if "-" in tgt:
                    name = "-".join(tgt.split("-")[:-1])
                else:
                    name = tgt
                G.add_node(next_id, name=d_clean(name))
                next_id += 1

            G.add_edge(char_to_id[src], char_to_id[tgt])
            G[char_to_id[src]][char_to_id[tgt]].update({edge_attr: edge})

    return G, root_id
