import json
from collections import defaultdict
from graphviz import Source
import networkx as nx
import pandas as pd
from networkx.algorithms.isomorphism import DiGraphMatcher
from sklearn.metrics import precision_recall_fscore_support
from tuw_nlp.graph.utils import GraphFormulaMatcher, pn_to_graph


def one_versus_rest(df, entity):
    mapper = {entity: 1}

    one_versus_rest_df = df.copy()
    one_versus_rest_df["one_versus_rest"] = [
        mapper[item] if item in mapper else 0 for item in df.label]

    return one_versus_rest_df


def train_feature(cl, feature, data):
    feature_graph = pn_to_graph(feature)[0]
    graphs = data.graph.tolist()
    labels = one_versus_rest(data, cl).one_versus_rest.tolist()
    path = "trained_features.tsv"
    with open(path, "w+") as f:
        for i, g in enumerate(graphs):
            matcher = DiGraphMatcher(
                g, feature_graph, node_match=GraphFormulaMatcher.node_matcher, edge_match=GraphFormulaMatcher.edge_matcher)
            if matcher.subgraph_is_isomorphic():
                for iso_pairs in matcher.subgraph_isomorphisms_iter():
                    nodes = []
                    for k in iso_pairs:
                        if feature_graph.nodes[iso_pairs[k]]["name"] == ".*":
                            nodes.append(g.nodes[k]["name"])
                    nodes_str = ",".join(nodes)
                    label = labels[i]
                    sentence = data.iloc[i].sentence
                    f.write(
                        f"{feature}\t{nodes_str}\t{sentence}\t{label}\n")

    return path


def cluster_feature(path):

    def to_dot(graph, feature):
        lines = [u'digraph finite_state_machine {']
        lines.append('\tdpi=70;label=' + '"' + feature + '"')
        # lines.append('\tordering=out;')
        # sorting everything to make the process deterministic
        node_lines = []
        node_to_name = {}
        for node, n_data in graph.nodes(data=True):
            printname = node
            if 'color' in n_data and n_data['color'] == "red":
                node_line = u'\t{0} [shape = circle, label = "{1}", \
                        style=filled, fillcolor=red];'.format(
                    printname, printname.split("_")[0]).replace('-', '_')
            if 'color' in n_data and n_data['color'] == "green":
                node_line = u'\t{0} [shape = circle, label = "{1}", \
                        style="filled", fillcolor=green];'.format(
                    printname, printname.split("_")[0]).replace('-', '_')
            node_lines.append(node_line)
        lines += sorted(node_lines)

        edge_lines = []
        for u, v, edata in graph.edges(data=True):
            if 'color' in edata:
                edge_lines.append(
                    u'\t{0} -> {1} [ label = "{2}" ];'.format(u, v, edata['color']))

        lines += sorted(edge_lines)
        lines.append('}')
        return u'\n'.join(lines)

    with open("longman_zero_paths_one_exp.json") as f:
        graphs = json.load(f)

    words = {}
    with open("trained_features.tsv") as f:
        for line in f:
            fields = line.strip("\n").split("\t")
            words[fields[1] + "_" + fields[3]] = int(fields[3])
            feature = fields[0]
    graph = nx.MultiDiGraph()

    color_map = []
    for word in words:
        if words[word] == 1:
            color = "green"
        else:
            color = "red"
        graph.add_node(word, color=color)
        word_clean = word.split("_")[0]
        if word_clean in graphs:
            hypernyms = graphs[word_clean]
            for hypernym in hypernyms:
                hypernym_words = hypernyms[hypernym]
                for w in hypernym_words:
                    if hypernym == "1":
                        graph.add_edge(word, w, color=hypernym)

    d = Source(to_dot(graph, feature))
    d.engine = "circo"
    d.format = "png"
    
    return d.render(view=True)

def evaluate_feature(cl, features, data):
    measure_features = []
    graphs = data.graph.tolist()
    labels = one_versus_rest(data, cl).one_versus_rest.tolist()

    whole_predicted = []
    matched = defaultdict(list)
    matcher = GraphFormulaMatcher(features)
    for i, g in enumerate(graphs):
        feats = matcher.match(g)
        label = 0
        for key, feature in feats:
            matched[i].append(features[feature][0])
            label = 1
        whole_predicted.append(label)

    accuracy = []
    for pcf in precision_recall_fscore_support(labels, whole_predicted, average=None):
        accuracy.append(pcf[1])

    for feat in features:
        measure = [feat[0]]
        false_pos_g = []
        false_pos_s = []
        true_pos_g = []
        true_pos_s = []
        false_neg_g = []
        false_neg_s = []
        predicted = []
        for i, g in enumerate(graphs):
            feats = matched[i]
            label = 1 if feat[0] in feats else 0
            if label == 1 and labels[i] == 0:
                false_pos_g.append(g)
                sen = data.iloc[i].sentence
                e1 = data.iloc[i].e1
                e2 = data.iloc[i].e2
                lab = data.iloc[i].label
                false_pos_s.append((sen, e1, e2, lab))
            if label == 1 and labels[i] == 1:
                true_pos_g.append(g)
                sen = data.iloc[i].sentence
                e1 = data.iloc[i].e1
                e2 = data.iloc[i].e2
                lab = data.iloc[i].label
                true_pos_s.append((sen, e1, e2, lab))
            if label == 0 and labels[i] == 1:
                false_neg_g.append(g)
                sen = data.iloc[i].sentence
                e1 = data.iloc[i].e1
                e2 = data.iloc[i].e2
                lab = data.iloc[i].label
                false_neg_s.append((sen, e1, e2, lab))
            predicted.append(label)
        for pcf in precision_recall_fscore_support(labels, predicted, average=None):
            measure.append(pcf[1])
        measure.append(false_pos_g)
        measure.append(false_pos_s)
        measure.append(true_pos_g)
        measure.append(true_pos_s)
        measure.append(false_neg_g)
        measure.append(false_neg_s)
        measure_features.append(measure)

    df = pd.DataFrame(measure_features, columns=[
                      'Feature', 'Precision', 'Recall', "Fscore", "Support", "False_positive_graphs", "False_positive_sens", "True_positive_graphs", "True_positive_sens", "False_negative_graphs", "False_negative_sens"])

    return df, accuracy
