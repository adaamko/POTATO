import re

import numpy as np
from sklearn.tree import _tree


def tree_to_code(tree, feature_graph_strings, inverse_relabel):
    tree_ = tree.tree_
    feature_name = [
        feature_graph_strings[inverse_relabel[int(i)]]
        if i != _tree.TREE_UNDEFINED
        else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            for path in recurse(tree_.children_left[node], depth + 1):
                yield [path[0], path[1] + [name], path[2]]
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            for path in recurse(tree_.children_right[node], depth + 1):
                yield [path[0] + [name], path[1], path[2]]
        else:
            yield [[], [], np.argmax(tree_.value[node])]
            print("{}return {}".format(indent, np.argmax(tree_.value[node])))

    return recurse(0, 1)


def d_clean(string):
    s = string
    for c in "\\=@-,'\".!:;<>/{}[]()#^?":
        s = s.replace(c, "_")
    s = s.replace("$", "_dollars")
    s = s.replace("%", "_percent")
    s = s.replace("|", " ")
    s = s.replace("*", " ")
    if s == "#":
        s = "_number"
    keywords = ("graph", "node", "strict", "edge")
    if re.match("^[0-9]", s) or s in keywords:
        s = "X" + s
    return s


def to_dots(graphs, marked_nodes=set(), integ=False):
    lines = [u"digraph finite_state_machine {", "\tdpi=70;"]
    # lines.append('\tordering=out;')
    # sorting everything to make the process deterministic
    for i, graph in enumerate(graphs):
        s = "subgraph cluster_" + chr(ord("@") + i + 1) + " {"
        node_lines = []

        node_lines.append(s)
        node_to_name = {}
        for node, n_data in graph.nodes(data=True):
            if integ:
                d_node = d_clean(str(node))
            else:
                d_node = d_clean(n_data["name"])
            printname = d_node
            node_to_name[node] = printname
            if (
                "expanded" in n_data
                and n_data["expanded"]
                and printname in marked_nodes
            ):
                node_line = u'\t{0} [shape = circle, label = "{1}", \
                        style=filled, fillcolor=purple];'.format(
                    d_node, printname
                ).replace(
                    "-", "_"
                )
            elif "expanded" in n_data and n_data["expanded"]:
                node_line = u'\t{0} [shape = circle, label = "{1}", \
                        style="filled"];'.format(
                    d_node, printname
                ).replace(
                    "-", "_"
                )
            elif "fourlang" in n_data and n_data["fourlang"]:
                node_line = u'\t{0} [shape = circle, label = "{1}", \
                        style="filled", fillcolor=red];'.format(
                    d_node, printname
                ).replace(
                    "-", "_"
                )
            elif "substituted" in n_data and n_data["substituted"]:
                node_line = u'\t{0} [shape = circle, label = "{1}", \
                        style="filled"];'.format(
                    d_node, printname
                ).replace(
                    "-", "_"
                )
            elif printname in marked_nodes:
                node_line = u'\t{0} [shape = circle, label = "{1}", style=filled, fillcolor=lightblue];'.format(
                    d_node, printname
                ).replace(
                    "-", "_"
                )
            else:
                node_line = u'\t{0} [shape = circle, label = "{1}"];'.format(
                    d_node, printname
                ).replace("-", "_")
            node_lines.append(node_line)
        lines += sorted(node_lines)

        edge_lines = []
        for u, v, edata in graph.edges(data=True):
            if "color" in edata:
                d_node1 = node_to_name[u]
                d_node2 = node_to_name[v]
                edge_lines.append(
                    u'\t{0} -> {1} [ label = "{2}" ];'.format(
                        d_node1, d_node2, edata["color"]
                    )
                )

        lines += sorted(edge_lines)
        lines.append("}")
    lines.append("}")
    return u"\n".join(lines)


def to_dot(graph, marked_nodes=set(), integ=False):
    lines = [u"digraph finite_state_machine {", "\tdpi=70;"]
    # lines.append('\tordering=out;')
    # sorting everything to make the process deterministic
    node_lines = []
    node_to_name = {}
    for node, n_data in graph.nodes(data=True):
        if integ:
            d_node = d_clean(str(node))
        else:
            d_node = d_clean(n_data["name"])
        printname = d_node
        node_to_name[node] = printname
        if "expanded" in n_data and n_data["expanded"] and printname in marked_nodes:
            node_line = u'\t{0} [shape = circle, label = "{1}", \
                    style=filled, fillcolor=purple];'.format(
                d_node, printname
            ).replace(
                "-", "_"
            )
        elif "expanded" in n_data and n_data["expanded"]:
            node_line = u'\t{0} [shape = circle, label = "{1}", \
                    style="filled"];'.format(
                d_node, printname
            ).replace(
                "-", "_"
            )
        elif "fourlang" in n_data and n_data["fourlang"]:
            node_line = u'\t{0} [shape = circle, label = "{1}", \
                    style="filled", fillcolor=red];'.format(
                d_node, printname
            ).replace(
                "-", "_"
            )
        elif "substituted" in n_data and n_data["substituted"]:
            node_line = u'\t{0} [shape = circle, label = "{1}", \
                    style="filled"];'.format(
                d_node, printname
            ).replace(
                "-", "_"
            )
        elif printname in marked_nodes:
            node_line = u'\t{0} [shape = circle, label = "{1}", style=filled, fillcolor=lightblue];'.format(
                d_node, printname
            ).replace(
                "-", "_"
            )
        else:
            node_line = u'\t{0} [shape = circle, label = "{1}"];'.format(
                d_node, printname
            ).replace("-", "_")
        node_lines.append(node_line)
    lines += sorted(node_lines)

    edge_lines = []
    for u, v, edata in graph.edges(data=True):
        if "color" in edata:
            d_node1 = node_to_name[u]
            d_node2 = node_to_name[v]
            edge_lines.append(
                u'\t{0} -> {1} [ label = "{2}" ];'.format(
                    d_node1, d_node2, edata["color"]
                )
            )

    lines += sorted(edge_lines)
    lines.append("}")
    return u"\n".join(lines)
