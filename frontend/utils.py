import copy
import json
import re
import sys

import pandas as pd
import streamlit as st
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME

from potato.dataset.utils import default_pn_to_graph
from potato.graph_extractor.extract import FeatureEvaluator
from potato.models.trainer import GraphTrainer

from contextlib import contextmanager
from io import StringIO
from threading import current_thread


def init_session_states():
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
    if "df_statistics" not in st.session_state:
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
    if "min_edge" not in st.session_state:
        st.session_state.min_edge = 0

    if "rows_to_delete" not in st.session_state:
        st.session_state.rows_to_delete = []
    if "rls_after_delete" not in st.session_state:
        st.session_state.rls_after_delete = []

    if "df_to_train" not in st.session_state:
        st.session_state.df_to_train = pd.DataFrame

    if "applied_rules" not in st.session_state:
        st.session_state.applied_rules = []

    if "rank" not in st.session_state:
        st.session_state.rank = False


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


def save_after_modify(hand_made_rules, classes):
    st.session_state.features[classes] = copy.deepcopy(
        st.session_state.rls_after_delete
    )
    st.session_state.feature_df = get_df_from_rules(
        [";".join(feat[0]) for feat in st.session_state.features[classes]],
        [";".join(feat[1]) for feat in st.session_state.features[classes]],
    )
    save_rules = hand_made_rules or "saved_features.json"
    save_ruleset(save_rules, st.session_state.features)
    st.session_state.rows_to_delete = []
    rerun()


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


def save_dataframe(data, path):
    data.to_pickle(path)


@st.cache(allow_output_mutation=True)
def read_val(path):
    return pd.read_pickle(path)


def train_df(df, min_edge=0, rank=False):
    with st_stdout("code"):
        trainer = GraphTrainer(df)
        features = trainer.prepare_and_train(min_edge=min_edge, rank=rank)

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
            st.session_state.df.at[i, "label"] = st.session_state.inverse_labels[1]
            st.session_state.df.at[i, "applied_rules"] = pred
        else:
            st.session_state.df.at[i, "applied_rules"] = []
            if st.session_state.df.loc[i, "annotated"] == False:
                st.session_state.df.at[i, "label"] = ""


def show_ml_feature(classes, hand_made_rules):
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
        if st.session_state.features[classes]:
            st.session_state.feature_df = get_df_from_rules(
                [";".join(feat[0]) for feat in st.session_state.features[classes]],
                [";".join(feat[1]) for feat in st.session_state.features[classes]],
            )
            save_rules = hand_made_rules or "saved_features.json"
            save_ruleset(save_rules, st.session_state.features)
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

    if type == "FP":
        st.session_state.false_graph_number = graph_type[type]
    elif type == "TP":
        st.session_state.true_graph_number = graph_type[type]
    elif type == "FN":
        st.session_state.false_neg_number = graph_type[type]


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


def rank_and_suggest(classes, data, evaluator, rank_false_negatives=True):
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
        else:
            st.warning("Dataset is not evaluated!")
