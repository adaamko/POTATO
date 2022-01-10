import copy
import json
import re
import sys

from collections import defaultdict
from graphviz import Source
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import streamlit as st
import penman
import torch
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME

from xpotato.dataset.utils import default_pn_to_graph
from xpotato.graph_extractor.extract import FeatureEvaluator, GraphExtractor
from xpotato.models.trainer import GraphTrainer
from xpotato.dataset.utils import default_pn_to_graph
from tuw_nlp.graph.utils import GraphFormulaMatcher, graph_to_pn

from contextlib import contextmanager
from io import StringIO
from threading import current_thread


def rerun():
    raise st.experimental_rerun()


@st.cache(
    hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()},
    allow_output_mutation=True,
)
def match_texts(text_input, extractor, graph_format):
    texts = text_input.split("\n")
    feature_values = []
    for k in st.session_state.features:
        for f in st.session_state.features[k]:
            feature_values.append(f)
    matcher = GraphFormulaMatcher(feature_values, converter=default_pn_to_graph)

    graphs = list(extractor.parse_iterable([text for text in texts], graph_format))

    predicted = []

    for i, g in enumerate(graphs):
        feats = matcher.match(g)
        label = "NONE"
        pattern = None
        for key, feature in feats:
            label = key
            pattern = feature
        predicted.append(
            (label, feature_values[pattern][0] if pattern is not None else None)
        )

    return graphs, predicted


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


def get_df_from_rules(rules, negated_rules, classes=None):
    if classes:
        data = {
            "rules": rules,
            "negated_rules": negated_rules,
            "predicted_label": classes,
        }
    else:
        data = {"rules": rules, "negated_rules": negated_rules}
    df = pd.DataFrame(data)

    return df


def save_after_modify(hand_made_rules, classes=None):
    if classes:
        st.session_state.features[classes] = copy.deepcopy(
            st.session_state.rls_after_delete
        )
        st.session_state.feature_df = get_df_from_rules(
            [";".join(feat[0]) for feat in st.session_state.features[classes]],
            [";".join(feat[1]) for feat in st.session_state.features[classes]],
        )
    else:
        features_temp = defaultdict(list)
        for k in st.session_state.rls_after_delete:
            features_temp[k[2]].append(copy.deepcopy(k))

        st.session_state.features = features_temp

        features_merged = []
        for i in st.session_state.features:
            for j in st.session_state.features[i]:
                features_merged.append(copy.deepcopy(j))

        st.session_state.feature_df = get_df_from_rules(
            [";".join(feat[0]) for feat in features_merged],
            [";".join(feat[1]) for feat in features_merged],
            [feat[2] for feat in features_merged],
        )

    save_rules = hand_made_rules or "saved_features.json"
    save_ruleset(save_rules, st.session_state.features)
    st.session_state.rows_to_delete = []
    rerun()


def filter_label(df, label):
    df["label"] = df.apply(lambda x: label if label in x["labels"] else "NOT", axis=1)
    df["label_id"] = df.apply(lambda x: 0 if x["label"] == "NOT" else 1, axis=1)


@st.cache(allow_output_mutation=True)
def read_train(path, label=None):
    df = pd.read_pickle(path)
    if label is not None:
        filter_label(df, label)
    return df


def save_dataframe(data, path):
    data.to_pickle(path)


@st.cache(allow_output_mutation=True)
def read_val(path, label=None):
    df = pd.read_pickle(path)
    if label is not None:
        filter_label(df, label)
    return df


def train_df(df, min_edge=0, rank=False):
    with st_stdout("code"):
        trainer = GraphTrainer(df)
        features = trainer.prepare_and_train(min_edge=min_edge, rank=rank)

        return features


def rule_chooser():
    option = st.selectbox("Choose from the rules", st.session_state.sens)
    G, _ = default_pn_to_graph(option.split(";")[0])
    text_G, _ = default_pn_to_graph(option.split(";")[0])
    dot_graph = to_dot(text_G)
    st.graphviz_chart(dot_graph, use_container_width=True)
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
    """This function shows an AgGrid dataframe with the top10 features ranked.
    It also shows the precision, recall and f1-score of the rules along with the number of the TP and FP samples retrieved.

    Args:
        classes (string): The chosen class.
        hand_made_rules (string): Where to save the rules.
    """
    features = [feat[0][0][0] for feat in st.session_state.ml_feature]
    precisions = [f"{feat[1]:.3f}" for feat in st.session_state.ml_feature]
    recalls = [f"{feat[2]:.3f}" for feat in st.session_state.ml_feature]
    fscores = [f"{feat[3]:.3f}" for feat in st.session_state.ml_feature]

    true_positive_samples = [feat[5] for feat in st.session_state.ml_feature]
    false_positive_samples = [feat[6] for feat in st.session_state.ml_feature]

    ranked_df = pd.DataFrame(
        {
            "feature": features,
            "precision": precisions,
            "recall": recalls,
            "fscore": fscores,
            "TP": true_positive_samples,
            "FP": false_positive_samples,
        }
    )
    with st.form("rule_chooser") as f:
        st.markdown(
            """
        ##### Inspect rules
        __Tick to box next to the rules you want to accept, then click on the _accept_rules_ button.__
        
        __Unaccepted rules will be deleted__."""
        )
        gb = GridOptionsBuilder.from_dataframe(ranked_df)
        gb.configure_default_column(
            editable=False,
            resizable=True,
            sorteable=True,
            wrapText=True,
            autoHeight=True,
        )
        gb.configure_selection(
            "multiple",
            use_checkbox=True,
            groupSelectsChildren=True,
            groupSelectsFiltered=True,
        )

        go = gb.build()
        rule_grid = AgGrid(
            ranked_df,
            gridOptions=go,
            allow_unsafe_jscode=True,
            reload_data=False,
            update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.VALUE_CHANGED,
            theme="light",
            fit_columns_on_grid_load=False,
        )

        accept_rules = st.form_submit_button(label="accept_rules")

    if accept_rules:
        selected_rules = (
            rule_grid["selected_rows"]
            if rule_grid["selected_rows"]
            else rule_grid["data"].to_dict(orient="records")
        )
        for rule in selected_rules:
            st.session_state.features[classes].append([[rule["feature"]], [], classes])

        st.session_state.ml_feature = None
        if st.session_state.features[classes]:
            st.session_state.feature_df = get_df_from_rules(
                [";".join(feat[0]) for feat in st.session_state.features[classes]],
                [";".join(feat[1]) for feat in st.session_state.features[classes]],
            )
            save_rules = hand_made_rules or "saved_features.json"
            save_ruleset(save_rules, st.session_state.features)
            rerun()


def extract_data_from_dataframe(option):
    fp_graphs = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].False_positive_graphs
    fp_sentences = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].False_positive_sens
    fp_indices = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].False_positive_indices
    tp_graphs = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].True_positive_graphs
    tp_sentences = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].True_positive_sens
    tp_indices = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].True_positive_indices
    fn_graphs = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].False_negative_graphs
    fn_sentences = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].False_negative_sens
    fn_indices = st.session_state.df_statistics.iloc[
        st.session_state.sens.index(option)
    ].False_negative_indices
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
        fn_indices,
        fp_graphs,
        fp_sentences,
        fp_indices,
        fscore,
        prec,
        predicted,
        recall,
        support,
        tp_graphs,
        tp_sentences,
        tp_indices,
    )


def graph_viewer(type, graphs, sentences, ids, nodes):
    st.markdown(
        """
    Tick the box next to the graphs you want to see. 
    The rule that applied will be highlighted in the graph. 

    The penman format of the graph will be also shown, you can copy any of the part directly from the penman format if you want to add a new rule.
    """
    )
    df = pd.DataFrame(
        {
            "id": ids,
            "sentence": [sen[0] for sen in sentences],
            "label": [sen[1] for sen in sentences],
            "graph": [i for i in range(len(graphs))],
        }
    )
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        editable=True,
        resizable=True,
        sorteable=True,
        wrapText=True,
        autoHeight=True,
    )
    gb.configure_column("graph", hide=True)
    gb.configure_column("label", hide=True)
    gb.configure_selection(
        "single",
        use_checkbox=True,
        groupSelectsChildren=True,
        groupSelectsFiltered=True,
    )
    go = gb.build()
    selected_df = AgGrid(
        df,
        gridOptions=go,
        allow_unsafe_jscode=True,
        reload_data=False,
        update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.VALUE_CHANGED,
        width="100%",
        theme="light",
        fit_columns_on_grid_load=True,
    )

    if selected_df["selected_rows"]:
        sel_row = selected_df["selected_rows"][0]
        st.markdown(
            f"<span><b>Sentence:</b> {sel_row['sentence']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<span><b>Sentence ID:</b> {sel_row['id']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<span><b>Gold label:</b> {sel_row['label']}</span>",
            unsafe_allow_html=True,
        )
        st.text(f"{type}: {len(graphs)}")
        current_graph = graphs[sel_row["graph"]]
        dot_current_graph = to_dot(
            current_graph,
            marked_nodes=set(nodes),
        )

        with st.expander("Graph dot source", expanded=False):
            st.write(dot_current_graph)
        if st.session_state.download:
            graph_pipe = Source(dot_current_graph).pipe(format="svg")
            st.download_button(
                label="Download graph as SVG",
                data=graph_pipe,
                file_name="graph.svg",
                mime="mage/svg+xml",
            )

        st.graphviz_chart(
            dot_current_graph,
            use_container_width=True,
        )

        st.write("Penman format:")
        st.text(penman.encode(penman.decode(graph_to_pn(current_graph)), indent=10))
        st.write("In one line format:")
        st.write(graph_to_pn(current_graph))


def add_rule_manually(classes, hand_made_rules):
    st.markdown(
        "If you want multiple rules to apply at once you can add __;__ between them"
    )
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
        f"<span><b>Or you can get suggestions by our ML by clicking on the button below!</b></span>",
        unsafe_allow_html=True,
    )


def rank_and_suggest(classes, data, evaluator):
    suggest_new_rule = st.button("suggest new rules")
    if suggest_new_rule:
        if (
            st.session_state.suggested_features
            and st.session_state.suggested_features[classes]
        ):
            features_to_rank = st.session_state.suggested_features[classes][:10]
            with st.spinner("Ranking rules..."):
                if not st.session_state.df_statistics.empty and st.session_state.sens:
                    features_ranked = evaluator.rank_features(
                        classes,
                        features_to_rank,
                        data,
                        st.session_state.df_statistics.iloc[0].False_negative_indices,
                    )
                else:
                    features_ranked = evaluator.rank_features(
                        classes,
                        features_to_rank,
                        data,
                        [],
                    )

            for feat in features_ranked:
                st.session_state.suggested_features[classes].remove(feat[0])

            st.session_state.ml_feature = features_ranked
        else:
            st.warning(
                "No suggestions available, maybe you don't have the dataset trained?"
            )


###############################################################################
# Init classes
@st.cache()
def init_evaluator():
    return FeatureEvaluator()


@st.cache(
    hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()},
    allow_output_mutation=True,
)
def init_extractor(lang, graph_format):
    extractor = GraphExtractor(lang=lang, cache_fn=f"{lang}_nlp_cache")

    if graph_format == "ud":
        extractor.init_nlp()
    elif graph_format == "amr":
        extractor.init_amr()

    return extractor


def init_session_states():
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

    if "download" not in st.session_state:
        st.session_state.download = False


###############################################################################
