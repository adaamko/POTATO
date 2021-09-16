import argparse
import configparser
import copy
import datetime
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from io import StringIO
from threading import current_thread

import networkx as nx
import pandas as pd
import penman as pn
import requests
import streamlit as st
import streamlit.components.v1 as components
from exprel.dataset.utils import default_pn_to_graph
from exprel.graph_extractor.extract import FeatureEvaluator
from exprel.models.trainer import GraphTrainer
from PIL import Image
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode, JsCode
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from tuw_nlp.grammar.text_to_4lang import TextTo4lang
from tuw_nlp.graph.fourlang import FourLang
from tuw_nlp.graph.utils import GraphFormulaMatcher, pn_to_graph, read_alto_output

if "false_graph_number" not in st.session_state:
    st.session_state.false_graph_number = 0
if "true_graph_number" not in st.session_state:
    st.session_state.true_graph_number = 0
if "false_neg_number" not in st.session_state:
    st.session_state.false_neg_number = 0
if "predicted_num" not in st.session_state:
    st.session_state.predicted_num = 0
if "whole_accuracy" not in st.session_state:
    st.session_state.whole_accuracy = []
if "dataframe" not in st.session_state:
    st.session_state.df_statistics = pd.DataFrame
if "val_dataframe" not in st.session_state:
    st.session_state.val_dataframe = pd.DataFrame
if "whole_accuracy_val" not in st.session_state:
    st.session_state.whole_accuracy_val = []
if "feature_df" not in st.session_state:
    st.session_state.feature_df = pd.DataFrame
if "clustered_words_path" not in st.session_state:
    st.session_state.clustered_words_path = None
if "features" not in st.session_state:
    st.session_state.features = {}
if "suggested_features" not in st.session_state:
    st.session_state.suggested_features = {}
if "trained" not in st.session_state:
    st.session_state.trained = False
if "ml_feature" not in st.session_state:
    st.session_state.ml_feature = None
if "sens" not in st.session_state:
    st.session_state.sens = []


def rerun():
    raise st.experimental_rerun()


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


def to_dot(graph, marked_nodes=set(), integ=False):
    lines = ["digraph finite_state_machine {", "\tdpi=70;"]
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
        if "expanded" in n_data and n_data["expanded"] and printname in marked_nodes:
            node_line = '\t{0} [shape = circle, label = "{1}", \
                    style=filled, fillcolor=purple];'.format(
                d_node, printname
            ).replace(
                "-", "_"
            )
        elif "expanded" in n_data and n_data["expanded"]:
            node_line = '\t{0} [shape = circle, label = "{1}", \
                    style="filled"];'.format(
                d_node, printname
            ).replace(
                "-", "_"
            )
        elif "fourlang" in n_data and n_data["fourlang"]:
            node_line = '\t{0} [shape = circle, label = "{1}", \
                    style="filled", fillcolor=red];'.format(
                d_node, printname
            ).replace(
                "-", "_"
            )
        elif "substituted" in n_data and n_data["substituted"]:
            node_line = '\t{0} [shape = circle, label = "{1}", \
                    style="filled"];'.format(
                d_node, printname
            ).replace(
                "-", "_"
            )
        elif printname in marked_nodes:
            node_line = '\t{0} [shape = circle, label = "{1}", style=filled, fillcolor=lightblue];'.format(
                d_node, printname
            ).replace(
                "-", "_"
            )
        else:
            node_line = '\t{0} [shape = circle, label = "{1}"];'.format(
                d_node, printname
            ).replace("-", "_")
        node_lines.append(node_line)
    lines += sorted(node_lines)

    edge_lines = []
    for u, v, edata in graph.edges(data=True):
        if "color" in edata:
            d_node1 = node_to_name[u].replace("-", "_")
            d_node2 = node_to_name[v].replace("-", "_")
            edge_lines.append(
                '\t{0} -> {1} [ label = "{2}" ];'.format(
                    d_node1, d_node2, edata["color"]
                )
            )

    lines += sorted(edge_lines)
    lines.append("}")
    return "\n".join(lines)


def save_ruleset(path, features):
    with open(path, "w+") as f:
        json.dump(features, f)


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


def get_df_from_rules(rules, negated_rules):
    data = {"rules": rules, "negated_rules": negated_rules}
    # Create DataFrame.
    df = pd.DataFrame(data)

    return df


@st.cache(allow_output_mutation=True)
def load_text_to_4lang():
    tfl = TextTo4lang("en", "en_nlp_cache")
    return tfl


@st.cache()
def init_evaluator():
    return FeatureEvaluator()


@st.cache(allow_output_mutation=True)
def read_train(path):
    return pd.read_pickle(path)


@st.cache(allow_output_mutation=True)
def read_val(path):
    return pd.read_pickle(path)


def train_df(df):
    with st_stdout("code"):
        trainer = GraphTrainer(df)
        features = trainer.prepare_and_train()

        return features


def rule_chooser():
    option = st.selectbox("Choose from the rules", st.session_state.sens)
    G, _ = default_pn_to_graph(option.split(";")[0])
    text_G, _ = default_pn_to_graph(option.split(";")[0])
    st.graphviz_chart(to_dot(text_G), use_container_width=True)
    nodes = [d_clean(n[1]["name"].split("_")[0]) for n in text_G.nodes(data=True)]
    return nodes, option


def annotate_df(predicted):
    for i, pred in enumerate(predicted):
        if pred:
            st.session_state.df.loc[i, "label"] = st.session_state.inverse_labels[1]
            st.session_state.df.loc[i, "applied_rules"] = pred


def show_ml_feature(classes):
    st.markdown(
        f"<span>Feature: {st.session_state.ml_feature[0]}, Precision: <b>{st.session_state.ml_feature[1]:.3f}</b>, \
                        Recall: <b>{st.session_state.ml_feature[2]:.3f}</b>, Fscore: <b>{st.session_state.ml_feature[3]:.3f}</b>, \
                            Support: <b>{st.session_state.ml_feature[4]}</b></span>",
        unsafe_allow_html=True,
    )
    accept_rule = st.button("Accept")
    decline_rule = st.button("Decline")
    if accept_rule:
        st.session_state.features[classes].append(st.session_state.ml_feature[0])
        st.session_state.ml_feature = None
        rerun()
    elif decline_rule:
        st.session_state.ml_feature = None
        rerun()


def extract_data_from_dataframe(option):
    fp_graphs = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].False_positive_graphs
    fp_sentences = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].False_positive_sens
    tp_graphs = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].True_positive_graphs
    tp_sentences = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].True_positive_sens
    fn_graphs = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].False_negative_graphs
    fn_sentences = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].False_negative_sens
    prec = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].Precision
    recall = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].Recall
    fscore = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].Fscore
    support = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].Support
    predicted = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].Predicted
    return (
        fn_graphs,
        fn_sentences,
        fp_graphs,
        fp_sentences,
        fscore,
        prec,
        predicted,
        recall,
        support,
        tp_graphs,
        tp_sentences,
    )


def graph_viewer(type, graphs, sentences, nodes):
    graph_type = {
        "FP": st.session_state.false_graph_number,
        "TP": st.session_state.true_graph_number,
        "FN": st.session_state.false_neg_number,
    }
    if st.button(f"Previous {type}"):
        graph_type[type] = max(0, graph_type[type] - 1)
    if st.button(f"Next {type}"):
        graph_type[type] = min(
            graph_type[type] + 1,
            len(graphs) - 1,
        )
    if graph_type[type] > len(graphs) - 1:
        graph_type[type] = 0
    st.markdown(
        f"<span><b>Sentence:</b> {sentences[graph_type[type]][0]}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span><b>Gold label:</b> {sentences[graph_type[type]][1]}</span>",
        unsafe_allow_html=True,
    )
    st.text(f"{type}: {len(graphs)}")
    current_graph = graphs[graph_type[type]]
    st.graphviz_chart(
        to_dot(
            current_graph,
            marked_nodes=set(nodes),
        ),
        use_container_width=True,
    )


def add_rule_manually(classes, hand_made_rules):
    text = st.text_area("You can add a new rule here manually")
    negated_text = st.text_area("You can modify the negated features here")
    agree = st.button("Add rule to the ruleset")
    if agree:
        if not negated_text.strip():
            negated_features = []
        else:
            negated_features = negated_text.split(";")
        st.session_state.features[classes].append([[text], negated_features, classes])
        if st.session_state.features[classes]:
            st.session_state.feature_df = get_df_from_rules(
                [";".join(feat[0]) for feat in st.session_state.features[classes]],
                [";".join(feat[1]) for feat in st.session_state.features[classes]],
            )
            save_rules = hand_made_rules or "saved_features.json"
            save_ruleset(save_rules, st.session_state.features)
            rerun()
    st.markdown(
        f"<span><b>Or get suggestions by our ML!</b></span>",
        unsafe_allow_html=True,
    )


def rank_and_suggest(classes, data, evaluator):
    suggest_new_rule = st.button("suggest new rules")
    if suggest_new_rule:
        if (
            not st.session_state.df_statistics.empty
            and st.session_state.sens
            and st.session_state.suggested_features[classes]
        ):
            features_to_rank = st.session_state.suggested_features[classes][:5]
            with st.spinner("Ranking rules..."):
                features_ranked = evaluator.rank_features(
                    classes,
                    features_to_rank,
                    data,
                    st.session_state.df_statistics.iloc[0].False_negative_indices,
                )
            suggested_feature = features_ranked[0]
            st.session_state.suggested_features[classes].remove(suggested_feature[0])

            st.session_state.ml_feature = suggested_feature


def supervised_mode(
    evaluator, data, val_data, graph_format, feature_path, hand_made_rules
):
    if hand_made_rules:
        with open(hand_made_rules) as f:
            st.session_state.features = json.load(f)

    if not feature_path and not st.session_state.trained:
        st.sidebar.title("Train your dataset!")
        show_app = st.sidebar.button("Train")
        if show_app:
            st.session_state.suggested_features = train_df(data)
            st.session_state.trained = True
            with st_stdout("success"):
                print("Success, your dataset is trained, wait for the app to load..")
                time.sleep(3)
                rerun()
        st.markdown(
            "<h3 style='text-align: center; color: black;'>Your dataset is shown below, click the train button to train your dataset!</h3>",
            unsafe_allow_html=True,
        )
        sample_df = AgGrid(data, width="100%", fit_columns_on_grid_load=True)

        st.write("label distribution:")
        st.bar_chart(data.groupby("label").size())

        st.write("sentence lenghts:")
        st.bar_chart(data.text.str.len())

        st.write("common words:")
        st.bar_chart(
            pd.Series(" ".join(data["text"]).lower().split()).value_counts()[:100]
        )

    if st.session_state.trained or feature_path:
        col1, col2 = st.columns(2)

        if (
            feature_path
            and os.path.exists(feature_path)
            and not st.session_state.suggested_features
        ):
            with open(feature_path) as f:
                st.session_state.suggested_features = json.load(f)

        if not st.session_state.features:
            for key in st.session_state.suggested_features:
                pop_len = (
                    5
                    if len(st.session_state.suggested_features[key]) > 5
                    else len(st.session_state.suggested_features[key]) - 1
                )
                st.session_state.features[key] = [
                    st.session_state.suggested_features[key].pop(0)
                    for _ in range(pop_len)
                ]

        col1.header("Rule to apply")

        col2.header("Graphs and results")

        # if graph_format == "fourlang":
        #    tfl = load_text_to_4lang()

        with col1:
            classes = st.selectbox(
                "Choose class", list(st.session_state.features.keys())
            )

            st.session_state.feature_df = get_df_from_rules(
                [";".join(feat[0]) for feat in st.session_state.features[classes]],
                [";".join(feat[1]) for feat in st.session_state.features[classes]],
            )

            with st.form("example form") as f:
                gb = GridOptionsBuilder.from_dataframe(st.session_state.feature_df)
                # make all columns editable
                gb.configure_columns(["rules", "negated_rules"], editable=True)
                gb.configure_selection(
                    "multiple",
                    use_checkbox=True,
                    groupSelectsChildren=True,
                    groupSelectsFiltered=True,
                    # ◙pre_selected_rows=[1,2]
                )
                go = gb.build()
                ag = AgGrid(
                    st.session_state.feature_df,
                    gridOptions=go,
                    key="grid1",
                    allow_unsafe_jscode=True,
                    reload_data=True,
                    update_mode=GridUpdateMode.MODEL_CHANGED
                    | GridUpdateMode.VALUE_CHANGED,
                    width="100%",
                    theme="material",
                    fit_columns_on_grid_load=True,
                )

                delete_or_train = st.radio(
                    "Delete or Train selected rules", ("none", "delete", "train")
                )
                submit = st.form_submit_button(label="save updates")
                evaluate = st.form_submit_button(label="evaluate selected")

            if evaluate:
                feature_list = []
                selected_rules = (
                    ag["selected_rows"]
                    if ag["selected_rows"]
                    else ag["data"].to_dict(orient="records")
                )
                for rule in selected_rules:
                    positive_rules = (
                        rule["rules"].split(";")
                        if "rules" in rule and rule["rules"].strip()
                        else []
                    )
                    negated_rules = (
                        rule["negated_rules"].split(";")
                        if "negated_rules" in rule and rule["negated_rules"].strip()
                        else []
                    )
                    feature_list.append(
                        [
                            positive_rules,
                            negated_rules,
                            classes,
                        ]
                    )
                st.session_state.sens = [";".join(feat[0]) for feat in feature_list]
                with st.spinner("Evaluating rules..."):
                    (
                        st.session_state.df_statistics,
                        st.session_state.whole_accuracy,
                    ) = evaluator.evaluate_feature(
                        classes, feature_list, data, graph_format
                    )
                    (
                        st.session_state.val_dataframe,
                        st.session_state.whole_accuracy_val,
                    ) = evaluator.evaluate_feature(
                        classes,
                        feature_list,
                        val_data,
                        graph_format,
                    )
                st.success("Done!")
                rerun()

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
                            [
                                rule.split(";"),
                                negated_list[i].strip().split(";"),
                                classes,
                            ]
                        )
                if rows_to_delete and delete:
                    for r in feature_list:
                        if ";".join(r[0]) not in rows_to_delete:
                            rls_after_delete.append(r)
                elif rows_to_delete and train:
                    rls_after_delete = copy.deepcopy(feature_list)
                    rule_to_train = rows_to_delete[0]
                    if ";" in rule_to_train or ".*" not in rule_to_train:
                        st.text("Only single and underspecified rules can be trained!")
                    else:
                        selected_words = evaluator.train_feature(
                            classes, rule_to_train, data, graph_format
                        )

                        for f in selected_words:
                            rls_after_delete.append([[f], [], classes])
                else:
                    rls_after_delete = copy.deepcopy(feature_list)

                if rls_after_delete:
                    st.session_state.features[classes] = copy.deepcopy(rls_after_delete)
                    st.session_state.feature_df = get_df_from_rules(
                        [
                            ";".join(feat[0])
                            for feat in st.session_state.features[classes]
                        ],
                        [
                            ";".join(feat[1])
                            for feat in st.session_state.features[classes]
                        ],
                    )
                    save_rules = hand_made_rules or "saved_features.json"
                    save_ruleset(save_rules, st.session_state.features)
                    rerun()

            add_rule_manually(classes, hand_made_rules)
            rank_and_suggest(classes, data, evaluator)

            if st.session_state.ml_feature:
                show_ml_feature(classes)

        with col2:
            if not st.session_state.df_statistics.empty and st.session_state.sens:
                if st.session_state.sens:
                    nodes, option = rule_chooser()
                st.markdown(
                    f"<span>Result of using all the rules: Precision: <b>{st.session_state.whole_accuracy[0]:.3f}</b>, \
                        Recall: <b>{st.session_state.whole_accuracy[1]:.3f}</b>, Fscore: <b>{st.session_state.whole_accuracy[2]:.3f}</b>, \
                            Support: <b>{st.session_state.whole_accuracy[3]}</b></span>",
                    unsafe_allow_html=True,
                )
                (
                    fn_graphs,
                    fn_sentences,
                    fp_graphs,
                    fp_sentences,
                    fscore,
                    prec,
                    predicted,
                    recall,
                    support,
                    tp_graphs,
                    tp_sentences,
                ) = extract_data_from_dataframe(option)

                st.markdown(
                    f"<span>The rule's result: Precision: <b>{prec:.3f}</b>, Recall: <b>{recall:.3f}</b>, \
                        Fscore: <b>{fscore:.3f}</b>, Support: <b>{support}</b></span>",
                    unsafe_allow_html=True,
                )

                with st.expander("Show validation data", expanded=False):
                    val_prec = st.session_state.val_dataframe.iloc[
                        st.session_state.sens.index(option)
                    ].Precision
                    val_recall = st.session_state.val_dataframe.iloc[
                        st.session_state.sens.index(option)
                    ].Recall
                    val_fscore = st.session_state.val_dataframe.iloc[
                        st.session_state.sens.index(option)
                    ].Fscore
                    val_support = st.session_state.val_dataframe.iloc[
                        st.session_state.sens.index(option)
                    ].Support
                    st.markdown(
                        f"<span>Result of using all the rules on the validation data: Precision: <b>{st.session_state.whole_accuracy_val[0]:.3f}</b>, \
                            Recall: <b>{st.session_state.whole_accuracy_val[1]:.3f}</b>, Fscore: <b>{st.session_state.whole_accuracy_val[2]:.3f}</b>, \
                                Support: <b>{st.session_state.whole_accuracy_val[3]}</b></span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<span>The rule's result on the validation data: Precision: <b>{val_prec:.3f}</b>, \
                            Recall: <b>{val_recall:.3f}</b>, Fscore: <b>{val_fscore:.3f}</b>, \
                                Support: <b>{val_support}</b></span>",
                        unsafe_allow_html=True,
                    )

                tp_fp_fn_choice = (
                    "True Positive graphs",
                    "False Positive graphs",
                    "False Negative graphs",
                )
                tp_fp_fn = st.selectbox(
                    "Select the graphs you want to view", tp_fp_fn_choice
                )
                if tp_fp_fn == "False Positive graphs":
                    if fp_graphs:
                        graph_viewer("FP", fp_graphs, fp_sentences, nodes)

                elif tp_fp_fn == "True Positive graphs":
                    if tp_graphs:
                        graph_viewer("TP", tp_graphs, tp_sentences, nodes)

                elif tp_fp_fn == "False Negative graphs":
                    if fn_graphs:
                        graph_viewer("FN", fn_graphs, fn_sentences, nodes)


def unsupervised_mode(evaluator, data, graph_format, feature_path, hand_made_rules):

    if hand_made_rules:
        with open(hand_made_rules) as f:
            st.session_state.features = json.load(f)
    if "df" not in st.session_state:
        st.session_state.df = data.copy()
        if "annotated" not in st.session_state.df:
            st.session_state.df["annotated"] = False
        if "applied_rules" not in st.session_state.df:
            st.session_state.df["applied_rules"] = [
                [] for _ in range(len(st.session_state.df))
            ]
        st.session_state.df.reset_index(level=0, inplace=True)

    if "applied_rules" not in st.session_state:
        st.session_state.applied_rules = []

    df_annotated = st.session_state.df[st.session_state.df.annotated == True][
        ["index", "text", "label", "applied_rules"]
    ]
    df_unannotated = st.session_state.df[st.session_state.df.annotated == False][
        ["index", "text", "label", "applied_rules"]
    ]

    if "labels" not in st.session_state:
        st.text("Before we start, please provide labels you want to train")
        user_input = st.text_input("label encoding", "NOT:0,OFF:1")

        st.session_state.labels = {
            label.split(":")[0]: int(label.split(":")[1])
            for label in user_input.split(",")
        }

        st.write(st.session_state.labels)
        st.session_state.inverse_labels = {
            v: k for (k, v) in st.session_state.labels.items()
        }
    else:
        st.markdown(
            f"<span><b>Annotate samples here:</b></span>",
            unsafe_allow_html=True,
        )

        if st.session_state.applied_rules:
            st.markdown(
                f"<span>Currently the following rules are applied:</span>",
                unsafe_allow_html=True,
            )
            st.write(st.session_state.applied_rules)
        with st.form("annotate form") as f:
            gb = GridOptionsBuilder.from_dataframe(df_unannotated)
            gb.configure_default_column(
                editable=True,
                resizable=True,
                sorteable=True,
                wrapText=True,
                autoHeight=False,
            )
            # make all columns editable
            gb.configure_selection(
                "multiple",
                use_checkbox=True,
                groupSelectsChildren=True,
                groupSelectsFiltered=True,
            )
            go = gb.build()
            ag = AgGrid(
                df_unannotated,
                gridOptions=go,
                key="grid2",
                allow_unsafe_jscode=True,
                reload_data=True,
                update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.VALUE_CHANGED,
                width="100%",
                theme="material",
                fit_columns_on_grid_load=True,
            )

            annotate = st.form_submit_button("Annotate")

        if annotate:
            if ag["selected_rows"]:
                for row in ag["selected_rows"]:
                    st.session_state.df.loc[
                        row["index"], "label"
                    ] = st.session_state.inverse_labels[1]
                    st.session_state.df.loc[row["index"], "annotated"] = True
                rerun()

        st.markdown(
            f"<span>Samples you have already annotated:</span>",
            unsafe_allow_html=True,
        )
        ag = AgGrid(
            df_annotated,
            key="grid3",
            allow_unsafe_jscode=True,
            reload_data=True,
            update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.VALUE_CHANGED,
            width="100%",
            theme="material",
            fit_columns_on_grid_load=True,
        )

        train = st.button("Train!")
        if train:
            st.session_state.trained = True
            df_to_train = st.session_state.df.copy()
            df_to_train["label"] = df_to_train["label"].apply(
                lambda x: st.session_state.inverse_labels[0] if not x else x
            )
            df_to_train["label_id"] = df_to_train["label"].apply(
                lambda x: st.session_state.labels[x]
            )
            st.session_state.suggested_features = train_df(df_to_train)
            st.session_state.df_statistics = pd.DataFrame

        col1, col2 = st.columns(2)

        if st.session_state.trained and st.session_state.suggested_features:
            with col1:

                if not st.session_state.features:
                    for key in st.session_state.suggested_features:
                        st.session_state.features[key] = [
                            st.session_state.suggested_features[key].pop(0)
                        ]

                classes = st.selectbox(
                    "Choose class", list(st.session_state.features.keys())
                )

                st.session_state.feature_df = get_df_from_rules(
                    [";".join(feat[0]) for feat in st.session_state.features[classes]],
                    [";".join(feat[1]) for feat in st.session_state.features[classes]],
                )

                with st.form("example form") as f:
                    gb = GridOptionsBuilder.from_dataframe(st.session_state.feature_df)
                    # make all columns editable
                    gb.configure_columns(["rules", "negated_rules"], editable=True)
                    gb.configure_selection(
                        "multiple",
                        use_checkbox=True,
                        groupSelectsChildren=True,
                        groupSelectsFiltered=True,
                        # ◙pre_selected_rows=[1,2]
                    )
                    go = gb.build()
                    ag = AgGrid(
                        st.session_state.feature_df,
                        gridOptions=go,
                        key="grid1",
                        allow_unsafe_jscode=True,
                        reload_data=True,
                        update_mode=GridUpdateMode.MODEL_CHANGED
                        | GridUpdateMode.VALUE_CHANGED,
                        width="100%",
                        theme="material",
                        fit_columns_on_grid_load=True,
                    )

                    delete_or_train = st.radio(
                        "Delete or Train selected rules", ("none", "delete", "train")
                    )
                    submit = st.form_submit_button(label="save updates")
                    evaluate = st.form_submit_button(label="evaluate selected")
                    annotate = st.form_submit_button(label="annotate based on selected")

                feature_list = []
                selected_rules = (
                    ag["selected_rows"]
                    if ag["selected_rows"]
                    else ag["data"].to_dict(orient="records")
                )
                for rule in selected_rules:
                    positive_rules = (
                        rule["rules"].split(";")
                        if "rules" in rule and rule["rules"].strip()
                        else []
                    )
                    negated_rules = (
                        rule["negated_rules"].split(";")
                        if "negated_rules" in rule and rule["negated_rules"].strip()
                        else []
                    )
                    feature_list.append(
                        [
                            positive_rules,
                            negated_rules,
                            classes,
                        ]
                    )

                if evaluate or annotate:
                    st.session_state.sens = [";".join(feat[0]) for feat in feature_list]
                    with st.spinner("Evaluating rules..."):
                        (
                            st.session_state.df_statistics,
                            st.session_state.whole_accuracy,
                        ) = evaluator.evaluate_feature(
                            classes, feature_list, st.session_state.df, graph_format
                        )

                    st.success("Done!")

                    if annotate:
                        predicted_rules = [[] for _ in range(len(st.session_state.df))]
                        st.session_state.applied_rules = st.session_state.sens
                        for j, opt in enumerate(st.session_state.sens):
                            predicted = st.session_state.df_statistics.iloc[j].Predicted

                            predicted_indices = [
                                i for i, pred in enumerate(predicted) if pred == 1
                            ]

                            for ind in predicted_indices:
                                predicted_rules[ind].append(opt)
                        annotate_df(predicted_rules)
                        st.session_state.trained = False

                    rerun()

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
                                [
                                    rule.split(";"),
                                    negated_list[i].strip().split(";"),
                                    classes,
                                ]
                            )
                    if rows_to_delete and delete:
                        for r in feature_list:
                            if ";".join(r[0]) not in rows_to_delete:
                                rls_after_delete.append(r)
                    elif rows_to_delete and train:
                        rls_after_delete = copy.deepcopy(feature_list)
                        rule_to_train = rows_to_delete[0]
                        if ";" in rule_to_train or ".*" not in rule_to_train:
                            st.text(
                                "Only single and underspecified rules can be trained!"
                            )
                        else:
                            selected_words = evaluator.train_feature(
                                classes,
                                rule_to_train,
                                st.session_state.df,
                                graph_format,
                            )

                            for f in selected_words:
                                rls_after_delete.append([[f], [], classes])
                    else:
                        rls_after_delete = copy.deepcopy(feature_list)

                    if rls_after_delete:
                        st.session_state.features[classes] = copy.deepcopy(
                            rls_after_delete
                        )
                        st.session_state.feature_df = get_df_from_rules(
                            [
                                ";".join(feat[0])
                                for feat in st.session_state.features[classes]
                            ],
                            [
                                ";".join(feat[1])
                                for feat in st.session_state.features[classes]
                            ],
                        )
                        save_rules = hand_made_rules or "saved_features.json"
                        save_ruleset(save_rules, st.session_state.features)
                        rerun()

                add_rule_manually(classes, hand_made_rules)
                rank_and_suggest(classes, st.session_state.df, evaluator)

                if st.session_state.ml_feature:
                    show_ml_feature(classes)
            with col2:
                if not st.session_state.df_statistics.empty and st.session_state.sens:
                    if st.session_state.sens:
                        nodes, option = rule_chooser()
                    st.markdown(
                        f"<span>Result of using all the rules: Precision: <b>{st.session_state.whole_accuracy[0]:.3f}</b>, \
                            Recall: <b>{st.session_state.whole_accuracy[1]:.3f}</b>, Fscore: <b>{st.session_state.whole_accuracy[2]:.3f}</b>, \
                                Support: <b>{st.session_state.whole_accuracy[3]}</b></span>",
                        unsafe_allow_html=True,
                    )
                    (
                        fn_graphs,
                        fn_sentences,
                        fp_graphs,
                        fp_sentences,
                        fscore,
                        prec,
                        predicted,
                        recall,
                        support,
                        tp_graphs,
                        tp_sentences,
                    ) = extract_data_from_dataframe(option)

                    st.markdown(
                        f"<span>The rule's result: Precision: <b>{prec:.3f}</b>, Recall: <b>{recall:.3f}</b>, \
                            Fscore: <b>{fscore:.3f}</b>, Support: <b>{support}</b></span>",
                        unsafe_allow_html=True,
                    )

                    tp_fp_fn_choice = (
                        "Predicted",
                        "True Positive graphs",
                        "False Positive graphs",
                        "False Negative graphs",
                    )
                    tp_fp_fn = st.selectbox(
                        "Select the option you want to view", tp_fp_fn_choice
                    )

                    current_graph = None
                    if tp_fp_fn == "Predicted":
                        predicted_inds = [
                            i for i, pred in enumerate(predicted) if pred == 1
                        ]
                        if st.button("Previous Predicted"):
                            st.session_state.predicted_num = max(
                                0, st.session_state.predicted_num - 1
                            )
                        if st.button("Next Predicted"):
                            st.session_state.predicted_num = min(
                                st.session_state.predicted_num + 1,
                                len(predicted_inds) - 1,
                            )

                        if st.session_state.predicted_num > len(predicted_inds) - 1:
                            st.session_state.predicted_inds = 0

                        st.markdown(
                            f"<span><b>Sentence:</b> {st.session_state.df.iloc[predicted_inds[st.session_state.predicted_num]].text}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<span><b>Gold label:</b> {st.session_state.df.iloc[predicted_inds[st.session_state.predicted_num]].label}</span>",
                            unsafe_allow_html=True,
                        )
                        st.text(f"Predicted: {len(predicted_inds)}")
                        current_graph = st.session_state.df.iloc[
                            predicted_inds[st.session_state.predicted_num]
                        ].graph
                        st.graphviz_chart(
                            to_dot(
                                current_graph,
                                marked_nodes=set(nodes),
                            ),
                            use_container_width=True,
                        )

                    elif tp_fp_fn == "False Positive graphs":
                        if fp_graphs:
                            graph_viewer("FP", fp_graphs, fp_sentences, nodes)

                    elif tp_fp_fn == "True Positive graphs":
                        if tp_graphs:
                            graph_viewer("TP", tp_graphs, tp_sentences, nodes)

                    elif tp_fp_fn == "False Negative graphs":
                        if fn_graphs:
                            graph_viewer("FN", fn_graphs, fn_sentences, nodes)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--train-data", type=str, required=True)
    parser.add_argument("-v", "--val-data", type=str)
    parser.add_argument("-sr", "--suggested-rules", default=None, type=str)
    parser.add_argument("-hr", "--hand-rules", default=None, type=str)
    parser.add_argument("-m", "--mode", default="supervised", type=str)
    parser.add_argument("-g", "--graph-format", default="fourlang", type=str)
    return parser.parse_args()


def main(args):
    st.set_page_config(layout="wide")
    st.markdown(
        "<h1 style='text-align: center; color: black;'>Rule extraction framework</h1>",
        unsafe_allow_html=True,
    )

    evaluator = init_evaluator()
    data = read_train(args.train_data)
    if args.val_data:
        val_data = read_val(args.val_data)
    graph_format = args.graph_format
    feature_path = args.suggested_rules
    hand_made_rules = args.hand_rules
    mode = args.mode
    if mode == "supervised":
        assert args.val_data
        supervised_mode(
            evaluator, data, val_data, graph_format, feature_path, hand_made_rules
        )
    elif mode == "unsupervised":
        unsupervised_mode(evaluator, data, graph_format, feature_path, hand_made_rules)


if __name__ == "__main__":
    args = get_args()
    main(args)
