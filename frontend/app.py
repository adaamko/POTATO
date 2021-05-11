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
from tuw_nlp.graph.utils import GraphFormulaMatcher, pn_to_graph, read_alto_output
from feature_evaluator import evaluate_feature

# SessionState module from https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
import SessionState

ruleset = SessionState.get(
    false_graph_number=0, true_graph_number=0, false_neg_number=0, whole_accuracy = [], rewritten_rules=[], negated_rules=[], dataframe=pd.DataFrame)


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

    col2.header("Graphs and results")

    data = pd.read_pickle("train_dataset")

    with col1:
        classes = st.selectbox("Choose label", list(features.keys()))
        sens = [";".join(feat[0]) for feat in features[classes]]
        option = "Rules to add here"
        option = st.selectbox(
            'Choose from the rules', sens)
        G, _ = read_alto_output(option.split(";")[0])
        nodes = [d_clean(n.split("_")[0]) for n in G.nodes()]

        text = st.text_area("You can modify the rule here", option)

        negated_features = ";".join(features[classes][sens.index(option)][1])

        negated_text = st.text_area("You can modify the negated features here", negated_features)

        evaluate = st.button("Evaluate ruleset")
        if evaluate:
            ruleset.dataframe, ruleset.whole_accuracy = evaluate_feature(
                classes, features[classes], data)

        text_G, _ = pn_to_graph(text.split(";")[0])
        st.graphviz_chart(
            to_dot(text_G), use_container_width=True)
        nodes = [d_clean(n[1]["name"].split("_")[0])
                 for n in text_G.nodes(data=True)]

        agree = st.button("Add rewritten rule to the ruleset")
        if agree:
            ruleset.rewritten_rules.append(text.split(";"))
            neg_rules = []
            for n in negated_text.split(";"):
                if n:
                    neg_rules.append(n)
            ruleset.negated_rules.append(neg_rules)

        if st.button('Remove the last rule from the set'):
            ruleset.rewritten_rules.pop()
            ruleset.negated_rules.pop()

        if st.button('Clear rules'):
            ruleset.rewritten_rules.clear()
            ruleset.negated_rules.clear()

        if st.button("Save rules"):
            features[classes] = [[rule, ruleset.negated_rules[i], classes]
                                 for i, rule in enumerate(ruleset.rewritten_rules)]
            save_ruleset("features.json", features)
            ruleset.rewritten_rules.clear()
            ruleset.negated_rules.clear()

        ruleset_expander = st.beta_expander(
            "Show the ruleset:", expanded=False)

        with ruleset_expander:
            if ruleset.rewritten_rules:
                for i, rule in enumerate(ruleset.rewritten_rules):
                    st.markdown(
                        f'<span style="color:red"><b>{rule}</b>; Negated rules: <b>{ruleset.negated_rules[i]}</b><br></span>', unsafe_allow_html=True)

    with col2:
        if not ruleset.dataframe.empty:
            st.markdown(
                f"<span>Result of using all the rules: Precision: <b>{ruleset.whole_accuracy[0]:.3f}</b>, Recall: <b>{ruleset.whole_accuracy[1]:.3f}</b>, Fscore: <b>{ruleset.whole_accuracy[2]:.3f}</b>, Support: <b>{ruleset.whole_accuracy[3]}</b></span>", unsafe_allow_html=True)

            fp_graphs = ruleset.dataframe.iloc[sens.index(
                option)].False_positive_graphs
            fp_sentences = ruleset.dataframe.iloc[sens.index(
                option)].False_positive_sens

            tp_graphs = ruleset.dataframe.iloc[sens.index(
                option)].True_positive_graphs
            tp_sentences = ruleset.dataframe.iloc[sens.index(
                option)].True_positive_sens

            fn_graphs = ruleset.dataframe.iloc[sens.index(
                option)].False_negative_graphs
            fn_sentences = ruleset.dataframe.iloc[sens.index(
                option)].False_negative_sens

            prec = ruleset.dataframe.iloc[sens.index(option)].Precision
            recall = ruleset.dataframe.iloc[sens.index(option)].Recall
            fscore = ruleset.dataframe.iloc[sens.index(option)].Fscore
            support = ruleset.dataframe.iloc[sens.index(option)].Support

            st.markdown(
                f"<span>The rule's result: Precision: <b>{prec:.3f}</b>, Recall: <b>{recall:.3f}</b>, Fscore: <b>{fscore:.3f}</b>, Support: <b>{support}</b></span>", unsafe_allow_html=True)

            tp_fp_fn_choice = ("True Positive graphs", "False Positive graphs", "False Negative graphs")
            tp_fp_fn = st.selectbox(
                'Select the graphs you want to view', tp_fp_fn_choice)

            if tp_fp_fn == "False Positive graphs":
                if fp_graphs:
                    if st.button("Previous FP"):
                        ruleset.false_graph_number = max(
                            0, ruleset.false_graph_number-1)
                    if st.button("Next FP"):
                        ruleset.false_graph_number = min(
                            ruleset.false_graph_number + 1, len(fp_graphs)-1)

                    if ruleset.false_graph_number > len(fp_graphs)-1:
                        ruleset.false_graph_number = 0

                    fp_graphs[ruleset.false_graph_number].remove_nodes_from(
                        list(nx.isolates(fp_graphs[ruleset.false_graph_number])))

                    st.markdown(
                        f"<span><b>Sentence:</b> {fp_sentences[ruleset.false_graph_number][0]}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<span><b>Entity1:</b> {fp_sentences[ruleset.false_graph_number][1]}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<span><b>Entity2:</b> {fp_sentences[ruleset.false_graph_number][2]}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<span><b>Gold label:</b> {fp_sentences[ruleset.false_graph_number][3]}</span>", unsafe_allow_html=True)
                    st.text(f"False positives: {len(fp_graphs)}")
                    st.graphviz_chart(
                        to_dot(fp_graphs[ruleset.false_graph_number], marked_nodes=set(nodes)), use_container_width=True)

            elif tp_fp_fn == "True Positive graphs":
                if tp_graphs:
                    if st.button("Previous TP"):
                        ruleset.true_graph_number = max(
                            0, ruleset.true_graph_number-1)
                    if st.button("Next TP"):
                        ruleset.true_graph_number = min(
                            ruleset.true_graph_number + 1, len(tp_graphs)-1)

                    if ruleset.true_graph_number > len(tp_graphs)-1:
                        ruleset.true_graph_number = 0

                    tp_graphs[ruleset.true_graph_number].remove_nodes_from(
                        list(nx.isolates(tp_graphs[ruleset.true_graph_number])))

                    st.markdown(
                        f"<span><b>Sentence:</b> {tp_sentences[ruleset.true_graph_number][0]}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<span><b>Entity1:</b> {tp_sentences[ruleset.true_graph_number][1]}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<span><b>Entity2:</b> {tp_sentences[ruleset.true_graph_number][2]}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<span><b>Gold label:</b> {tp_sentences[ruleset.true_graph_number][3]}</span>", unsafe_allow_html=True)
                    st.text(f"True positives: {len(tp_graphs)}")
                    st.graphviz_chart(
                        to_dot(tp_graphs[ruleset.true_graph_number], marked_nodes=set(nodes)), use_container_width=True)
            elif tp_fp_fn == "False Negative graphs":
                if fn_graphs:
                    if st.button("Previous FN"):
                        ruleset.false_neg_number = max(
                            0, ruleset.false_neg_number-1)
                    if st.button("Next FN"):
                        ruleset.false_neg_number = min(
                            ruleset.false_neg_number + 1, len(fn_graphs)-1)

                    if ruleset.false_neg_number > len(fn_graphs)-1:
                        ruleset.false_neg_number = 0

                    tp_graphs[ruleset.false_neg_number].remove_nodes_from(
                        list(nx.isolates(fn_graphs[ruleset.false_neg_number])))

                    st.markdown(
                        f"<span><b>Sentence:</b> {fn_sentences[ruleset.false_neg_number][0]}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<span><b>Entity1:</b> {fn_sentences[ruleset.false_neg_number][1]}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<span><b>Entity2:</b> {fn_sentences[ruleset.false_neg_number][2]}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<span><b>Gold label:</b> {fn_sentences[ruleset.false_neg_number][3]}</span>", unsafe_allow_html=True)
                    st.text(f"False negatives: {len(fn_graphs)}")
                    st.graphviz_chart(
                        to_dot(fn_graphs[ruleset.false_neg_number], marked_nodes=set(nodes)), use_container_width=True)


if __name__ == "__main__":
    main()
