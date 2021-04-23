import datetime
import json
import os
import random
import re
from collections import defaultdict

import networkx as nx
import pandas as pd
import penman as pn
import requests
import streamlit as st
import streamlit.components.v1 as components
from tuw_nlp.graph.utils import GraphMatcher, pn_to_graph, read_alto_output
from feature_evaluator import evaluate_feature

# SessionState module from https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
import SessionState

ruleset = SessionState.get(
    graph_number=0, rewritten_rules=[], dataframe=pd.DataFrame)


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
    return s


def to_dot(graph, marked_nodes=set(), integ=False):
    lines = [u'digraph finite_state_machine {', '\tdpi=70;']
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
        if 'expanded' in n_data and n_data['expanded'] and printname in marked_nodes:
            node_line = u'\t{0} [shape = circle, label = "{1}", \
                    style=filled, fillcolor=purple];'.format(
                d_node, printname).replace('-', '_')
        elif 'expanded' in n_data and n_data['expanded']:
            node_line = u'\t{0} [shape = circle, label = "{1}", \
                    style="filled"];'.format(
                d_node, printname).replace('-', '_')
        elif 'fourlang' in n_data and n_data['fourlang']:
            node_line = u'\t{0} [shape = circle, label = "{1}", \
                    style="filled", fillcolor=red];'.format(
                d_node, printname).replace('-', '_')
        elif 'substituted' in n_data and n_data['substituted']:
            node_line = u'\t{0} [shape = circle, label = "{1}", \
                    style="filled"];'.format(
                d_node, printname).replace('-', '_')
        elif printname in marked_nodes:
            node_line = u'\t{0} [shape = circle, label = "{1}", style=filled, fillcolor=lightblue];'.format(
                d_node, printname).replace('-', '_')
        else:
            node_line = u'\t{0} [shape = circle, label = "{1}"];'.format(
                d_node, printname).replace('-', '_')
        node_lines.append(node_line)
    lines += sorted(node_lines)

    edge_lines = []
    for u, v, edata in graph.edges(data=True):
        if 'color' in edata:
            d_node1 = node_to_name[u]
            d_node2 = node_to_name[v]
            edge_lines.append(
                u'\t{0} -> {1} [ label = "{2}" ];'.format(d_node1, d_node2, edata['color']))

    lines += sorted(edge_lines)
    lines.append('}')
    return u'\n'.join(lines)


def save_ruleset(path, features):
    with open(path, "w+") as f:
        json.dump(features, f)


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
    return s


def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; color: black;'>Rule extraction framework</h1>",
                unsafe_allow_html=True)
    col1, col2 = st.beta_columns(2)

    with open("features.json") as f:
        features = json.load(f)

    col1.header("Rule to apply")

    col2.header("False positive graphs generated")

    val_data = pd.read_pickle("validation_dataset")

    with col1:
        classes = st.selectbox("Choose label", list(features.keys()))
        sens = [feat[0] for feat in features[classes]]
        option = "Rules to add here"
        option = st.selectbox(
            'Choose from the rules', sens)
        G, _ = read_alto_output(option)
        nodes = [d_clean(n.split("_")[0]) for n in G.nodes()]

        text = st.text_area("You can modify the rule here", option)

        evaluate = st.button("Evaluate ruleset")
        if evaluate:
            ruleset.dataframe = evaluate_feature(
                classes, features[classes], val_data)

        text_G, _ = pn_to_graph(text)
        st.graphviz_chart(
            to_dot(text_G), use_container_width=True)
        nodes = [d_clean(n[1]["name"].split("_")[0])
                 for n in text_G.nodes(data=True)]

        agree = st.button("Add rewritten rule to the ruleset")
        if agree:
            ruleset.rewritten_rules.append(text)

        if st.button('Remove the last rule from the set'):
            ruleset.rewritten_rules.pop()

        if st.button('Clear rules'):
            ruleset.rewritten_rules.clear()

        if st.button("Save rules"):
            features[classes] = [[rule, classes]
                                 for rule in ruleset.rewritten_rules]
            save_ruleset("features.json", features)
            ruleset.rewritten_rules.clear()

        ruleset_expander = st.beta_expander(
            "Show the ruleset:", expanded=False)

        with ruleset_expander:
            rules = ", ".join(ruleset.rewritten_rules)
            st.markdown(
                f'<span style="color:red"><b>{rules}</b></span>', unsafe_allow_html=True)

    with col2:
        if not ruleset.dataframe.empty:
            graphs = ruleset.dataframe.iloc[sens.index(
                option)].False_positive_graphs
            sentences = ruleset.dataframe.iloc[sens.index(
                option)].False_positive_sens
            prec = ruleset.dataframe.iloc[sens.index(option)].Precision
            recall = ruleset.dataframe.iloc[sens.index(option)].Recall
            fscore = ruleset.dataframe.iloc[sens.index(option)].Fscore
            support = ruleset.dataframe.iloc[sens.index(option)].Support

            st.markdown(
                f"<span>The rule's result: Precision: <b>{prec:.3f}</b>, Recall: <b>{recall:.3f}</b>, Fscore: <b>{fscore:.3f}</b>, Support: <b>{support}</b></span>", unsafe_allow_html=True)

            if graphs:
                if st.button("Previous"):
                    ruleset.graph_number = max(0, ruleset.graph_number-1)
                if st.button("Next"):
                    ruleset.graph_number = min(
                        ruleset.graph_number + 1, len(graphs)-1)

                if ruleset.graph_number > len(graphs)-1:
                    ruleset.graph_number = 0

                graphs[ruleset.graph_number].remove_nodes_from(
                    list(nx.isolates(graphs[ruleset.graph_number])))

                st.markdown(
                    f"<span><b>Sentence:</b> {sentences[ruleset.graph_number][0]}</span>", unsafe_allow_html=True)
                st.markdown(
                    f"<span><b>Entity1:</b> {sentences[ruleset.graph_number][1]}</span>", unsafe_allow_html=True)
                st.markdown(
                    f"<span><b>Entity2:</b> {sentences[ruleset.graph_number][2]}</span>", unsafe_allow_html=True)
                st.graphviz_chart(
                    to_dot(graphs[ruleset.graph_number], marked_nodes=set(nodes)), use_container_width=True)


if __name__ == "__main__":
    main()
