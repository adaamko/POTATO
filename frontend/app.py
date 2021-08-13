import datetime
import json
import os
import random
import re
import configparser
from collections import defaultdict
import copy

import networkx as nx
import pandas as pd
import penman as pn
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from tuw_nlp.graph.fourlang import FourLang
from tuw_nlp.grammar.text_to_4lang import TextTo4lang
from tuw_nlp.graph.utils import (GraphFormulaMatcher, pn_to_graph,
                                 read_alto_output)
from exprel.dataset.utils import amr_pn_to_graph

# SessionState module from https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
from feature_evaluator import cluster_feature, evaluate_feature, train_feature
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

if "false_graph_number" not in st.session_state:
    st.session_state.false_graph_number = 0
if "true_graph_number" not in st.session_state:
    st.session_state.true_graph_number = 0
if "false_neg_number" not in st.session_state:
    st.session_state.false_neg_number = 0
if "whole_accuracy" not in st.session_state:
    st.session_state.whole_accuracy = []
if "dataframe" not in st.session_state:
    st.session_state.dataframe = pd.DataFrame
if "val_dataframe" not in st.session_state:
    st.session_state.val_dataframe = pd.DataFrame
if "whole_accuracy_val" not in st.session_state:
    st.session_state.whole_accuracy_val = []
if "feature_df" not in st.session_state:
    st.session_state.feature_df = pd.DataFrame
if "clustered_words_path" not in st.session_state:
    st.session_state.clustered_words_path = None


def rerun():
    raise st.experimental_rerun()


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
            d_node = d_clean(n_data["name"]) if n_data["name"] else "None"
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
            d_node1 = node_to_name[u].replace('-', '_')
            d_node2 = node_to_name[v].replace('-', '_')
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


def get_df_from_rules(rules, negated_rules):
    data = {'rules': rules, 'negated_rules': negated_rules}
    # Create DataFrame.
    df = pd.DataFrame(data)

    return df


@st.cache(allow_output_mutation=True)
def load_text_to_4lang():
    tfl = TextTo4lang("en", "en_nlp_cache")
    return tfl


@st.cache(allow_output_mutation=True)
def read_train(path):
    return pd.read_pickle(path)


@st.cache(allow_output_mutation=True)
def read_val(path):
    return pd.read_pickle(path)


def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; color: black;'>Rule extraction framework</h1>",
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    config = configparser.ConfigParser()
    config.read("app_config.ini")
    feature_path = config["DEFAULT"]["features_path"]
    train_path = config["DEFAULT"]["train_path"]
    val_path = config["DEFAULT"]["validation_path"]
    graph_format = config["DEFAULT"]["graph_format"]
    with open(feature_path) as f:
        features = json.load(f)

    col1.header("Rule to apply")

    col2.header("Graphs and results")

    data = read_train(train_path)
    val_data = read_val(val_path)

    if graph_format == "fourlang":
        tfl = load_text_to_4lang()

    with col1:
        classes = st.selectbox("Choose class", list(features.keys()))
        sens = [";".join(feat[0]) for feat in features[classes]]
        option = "Rules to add here"
        option = st.selectbox(
            'Choose from the rules', sens)

        if graph_format == "amr":
            G, _ = amr_pn_to_graph(option.split(";")[0])
        else:
            G, _ = read_alto_output(option.split(";")[0])

        if graph_format == "amr":
            text_G, _ = amr_pn_to_graph(option.split(";")[0])
        else:
            text_G, _ = pn_to_graph(option.split(";")[0])

        st.graphviz_chart(
            to_dot(text_G), use_container_width=True)
        nodes = [d_clean(n[1]["name"].split("_")[0])
                 for n in text_G.nodes(data=True)]

        text = st.text_area("You can add a new rule here")

        negated_text = st.text_area(
            "You can modify the negated features here")

        agree = st.button("Add rule to the ruleset")
        if agree:
            if not negated_text.strip():
                negated_features = []
            else:
                negated_features = negated_text.split(";")
            features[classes].append([[text], negated_features, classes])
            if features[classes]:
                st.session_state.feature_df = get_df_from_rules(
                    [";".join(feat[0]) for feat in features[classes]], [";".join(feat[1]) for feat in features[classes]])
                save_ruleset(feature_path, features)
                rerun()
        st.session_state.feature_df = get_df_from_rules(
            [";".join(feat[0]) for feat in features[classes]], [";".join(feat[1]) for feat in features[classes]])
        with st.form('example form') as f:
            gb = GridOptionsBuilder.from_dataframe(st.session_state.feature_df)
            # make all columns editable
            gb.configure_columns(["rules", "negated_rules"], editable=True)
            gb.configure_selection(
                "multiple", use_checkbox=True, groupSelectsChildren=True, groupSelectsFiltered=True)
            go = gb.build()
            ag = AgGrid(st.session_state.feature_df, gridOptions=go, key='grid1', allow_unsafe_jscode=True, reload_data=True, update_mode=GridUpdateMode.MODEL_CHANGED,
                        width='100%', fit_columns_on_grid_load=True)
            delete_or_train = st.radio(
                "Delete or Train selected rules", ('none', 'delete', 'train'))
            submit = st.form_submit_button(label="save updates")

        if submit:
            delete = delete_or_train == "delete"
            train = delete_or_train == "train"

            rows_to_delete = [r["rules"] for r in ag["selected_rows"]]
            rls_after_delete = []

            negated_list = ag["data"]["negated_rules"].tolist()
            feature_list = []
            for i, rule in enumerate(ag["data"]["rules"].tolist()):
                if not negated_list[i].strip():
                    feature_list.append([rule.split(";"), [], classes])
                else:
                    feature_list.append(
                        [rule.split(";"), negated_list[i].strip().split(";"), classes])
            if rows_to_delete and delete:
                for r in feature_list:
                    if ";".join(r[0]) not in rows_to_delete:
                        st.write(r)
                        rls_after_delete.append(r)
            elif rows_to_delete and train:
                rls_after_delete = copy.deepcopy(feature_list)
                rule_to_train = rows_to_delete[0]
                if ";" in rule_to_train or ".*" not in rule_to_train:
                    st.text("Only single and underspecified rules can be trained!")
                else:
                    trained_feature = train_feature(
                        classes, rule_to_train, data, graph_format)
                    st.session_state.clustered_words_path, selected_words = cluster_feature(
                        trained_feature)
                    for f in selected_words:
                        rls_after_delete.append([[f], [], classes])
            else:
                rls_after_delete = copy.deepcopy(feature_list)

            if rls_after_delete:
                features[classes] = copy.deepcopy(rls_after_delete)
                st.session_state.feature_df = get_df_from_rules(
                    [";".join(feat[0]) for feat in features[classes]], [";".join(feat[1]) for feat in features[classes]])
                save_ruleset(feature_path, features)
                rerun()

        evaluate = st.button("Evaluate ruleset")
        if evaluate:
            with st.spinner("Evaluating rules..."):
                st.session_state.dataframe, st.session_state.whole_accuracy = evaluate_feature(
                    classes, features[classes], data, graph_format)
                st.session_state.val_dataframe, st.session_state.whole_accuracy_val = evaluate_feature(
                    classes, features[classes], val_data, graph_format)
            st.success('Done!')

    with col2:
        if not st.session_state.dataframe.empty:
            st.markdown(
                f"<span>Result of using all the rules: Precision: <b>{st.session_state.whole_accuracy[0]:.3f}</b>, Recall: <b>{st.session_state.whole_accuracy[1]:.3f}</b>, Fscore: <b>{st.session_state.whole_accuracy[2]:.3f}</b>, Support: <b>{st.session_state.whole_accuracy[3]}</b></span>", unsafe_allow_html=True)

            fp_graphs = st.session_state.dataframe.iloc[sens.index(
                option)].False_positive_graphs
            fp_sentences = st.session_state.dataframe.iloc[sens.index(
                option)].False_positive_sens

            tp_graphs = st.session_state.dataframe.iloc[sens.index(
                option)].True_positive_graphs
            tp_sentences = st.session_state.dataframe.iloc[sens.index(
                option)].True_positive_sens

            fn_graphs = st.session_state.dataframe.iloc[sens.index(
                option)].False_negative_graphs
            fn_sentences = st.session_state.dataframe.iloc[sens.index(
                option)].False_negative_sens

            prec = st.session_state.dataframe.iloc[sens.index(
                option)].Precision
            recall = st.session_state.dataframe.iloc[sens.index(option)].Recall
            fscore = st.session_state.dataframe.iloc[sens.index(option)].Fscore
            support = st.session_state.dataframe.iloc[sens.index(
                option)].Support

            st.markdown(
                f"<span>The rule's result: Precision: <b>{prec:.3f}</b>, Recall: <b>{recall:.3f}</b>, Fscore: <b>{fscore:.3f}</b>, Support: <b>{support}</b></span>", unsafe_allow_html=True)

            with st.expander("Show validation data", expanded=False):
                val_prec = st.session_state.val_dataframe.iloc[sens.index(
                    option)].Precision
                val_recall = st.session_state.val_dataframe.iloc[sens.index(
                    option)].Recall
                val_fscore = st.session_state.val_dataframe.iloc[sens.index(
                    option)].Fscore
                val_support = st.session_state.val_dataframe.iloc[sens.index(
                    option)].Support
                st.markdown(
                    f"<span>Result of using all the rules on the validation data: Precision: <b>{st.session_state.whole_accuracy_val[0]:.3f}</b>, Recall: <b>{st.session_state.whole_accuracy_val[1]:.3f}</b>, Fscore: <b>{st.session_state.whole_accuracy_val[2]:.3f}</b>, Support: <b>{st.session_state.whole_accuracy_val[3]}</b></span>", unsafe_allow_html=True)
                st.markdown(
                    f"<span>The rule's result on the validation data: Precision: <b>{val_prec:.3f}</b>, Recall: <b>{val_recall:.3f}</b>, Fscore: <b>{val_fscore:.3f}</b>, Support: <b>{val_support}</b></span>", unsafe_allow_html=True)

            tp_fp_fn_choice = ("True Positive graphs",
                               "False Positive graphs", "False Negative graphs")
            tp_fp_fn = st.selectbox(
                'Select the graphs you want to view', tp_fp_fn_choice)

            current_graph = None
            if tp_fp_fn == "False Positive graphs":
                if fp_graphs:
                    if st.button("Previous FP"):
                        st.session_state.false_graph_number = max(
                            0, st.session_state.false_graph_number-1)
                    if st.button("Next FP"):
                        st.session_state.false_graph_number = min(
                            st.session_state.false_graph_number + 1, len(fp_graphs)-1)

                    if st.session_state.false_graph_number > len(fp_graphs)-1:
                        st.session_state.false_graph_number = 0

                    st.markdown(
                        f"<span><b>Sentence:</b> {fp_sentences[st.session_state.false_graph_number][0]}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<span><b>Gold label:</b> {fp_sentences[st.session_state.false_graph_number][1]}</span>", unsafe_allow_html=True)
                    st.text(f"False positives: {len(fp_graphs)}")
                    current_graph = fp_graphs[st.session_state.false_graph_number]
                    st.graphviz_chart(
                        to_dot(fp_graphs[st.session_state.false_graph_number], marked_nodes=set(nodes)), use_container_width=True)

            elif tp_fp_fn == "True Positive graphs":
                if tp_graphs:
                    if st.button("Previous TP"):
                        st.session_state.true_graph_number = max(
                            0, st.session_state.true_graph_number-1)
                    if st.button("Next TP"):
                        st.session_state.true_graph_number = min(
                            st.session_state.true_graph_number + 1, len(tp_graphs)-1)

                    if st.session_state.true_graph_number > len(tp_graphs)-1:
                        st.session_state.true_graph_number = 0

                    with open("graph.dot", "w+") as f:
                        f.write(
                            to_dot(tp_graphs[st.session_state.true_graph_number], marked_nodes=set(nodes)))

                    st.markdown(
                        f"<span><b>Sentence:</b> {tp_sentences[st.session_state.true_graph_number][0]}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<span><b>Gold label:</b> {tp_sentences[st.session_state.true_graph_number][1]}</span>", unsafe_allow_html=True)
                    st.text(f"True positives: {len(tp_graphs)}")
                    current_graph = tp_graphs[st.session_state.true_graph_number]
                    st.graphviz_chart(
                        to_dot(tp_graphs[st.session_state.true_graph_number], marked_nodes=set(nodes)), use_container_width=True)
            elif tp_fp_fn == "False Negative graphs":
                if fn_graphs:
                    if st.button("Previous FN"):
                        st.session_state.false_neg_number = max(
                            0, st.session_state.false_neg_number-1)
                    if st.button("Next FN"):
                        st.session_state.false_neg_number = min(
                            st.session_state.false_neg_number + 1, len(fn_graphs)-1)

                    if st.session_state.false_neg_number > len(fn_graphs)-1:
                        st.session_state.false_neg_number = 0

                    st.markdown(
                        f"<span><b>Sentence:</b> {fn_sentences[st.session_state.false_neg_number][0]}</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<span><b>Gold label:</b> {fn_sentences[st.session_state.false_neg_number][1]}</span>", unsafe_allow_html=True)
                    st.text(f"False negatives: {len(fn_graphs)}")
                    current_graph = fn_graphs[st.session_state.false_neg_number]
                    with open("graph.dot", "w+") as f:
                        f.write(to_dot(current_graph, marked_nodes=set(nodes)))
                    st.graphviz_chart(
                        to_dot(fn_graphs[st.session_state.false_neg_number], marked_nodes=set(nodes)), use_container_width=True)

            if graph_format == "fourlang":
                fl = FourLang(current_graph, 0)
                expand_node = st.text_input("Expand node", None)
                append_zero_path = st.button(
                    "Expand node and append zero paths to the graph")
                if append_zero_path:
                    tfl.expand(fl, depth=1, expand_set={
                               expand_node}, strategy="whitelisting")
                    fl.append_zero_paths()

                show_graph = st.expander(
                    "Show graph", expanded=False)

                with show_graph:
                    if current_graph:
                        st.graphviz_chart(
                            to_dot(fl.G, marked_nodes=set(nodes)), use_container_width=True)

                clustered_words = st.expander(
                    "Show clustered words:", expanded=False)

                with clustered_words:
                    if st.session_state.clustered_words_path:
                        image = Image.open(
                            st.session_state.clustered_words_path)
                        st.image(image, caption='trained_feature',
                                 use_column_width=True)


if __name__ == "__main__":
    main()
