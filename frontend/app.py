import argparse
import copy
import os
import time
import json
import streamlit as st
import pandas as pd
import penman
from graphviz import Source

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from tuw_nlp.graph.utils import graph_to_pn

from utils import (
    train_df,
    add_rule_manually,
    annotate_df,
    extract_data_from_dataframe,
    get_df_from_rules,
    graph_viewer,
    init_evaluator,
    init_extractor,
    init_session_states,
    rank_and_suggest,
    read_train,
    read_val,
    rerun,
    rule_chooser,
    save_ruleset,
    save_after_modify,
    save_dataframe,
    match_texts,
    show_ml_feature,
    st_stdout,
    to_dot,
)


def inference_mode(evaluator, hand_made_rules):
    st.sidebar.title("Inference Mode")
    st.sidebar.markdown(
        """
    ## Select the graph format
    """
    )
    graph_format = st.sidebar.selectbox("", ["ud", "fourlang", "amr"])
    st.sidebar.markdown(
        """
    ## Select the language
    """
    )
    lang = st.sidebar.selectbox("", ["en", "de"])
    if lang == "de" and graph_format == "amr":
        st.error("Currently AMR does not support German")
        return

    st.sidebar.markdown(
        """
        __Show download button for graphs (an option to allow downloading graphs)__
        """
    )
    st.session_state.download = st.sidebar.selectbox("", options=[False, True], key=2)

    if hand_made_rules:
        with open(hand_made_rules) as f:
            st.session_state.features = json.load(f)

    extractor = init_extractor(lang, graph_format)

    col1, col2 = st.columns(2)

    col1.header("Rule chooser and modifier")

    col2.header("Graph viewer and evaluator")

    with col1:
        features_merged = []
        for i in st.session_state.features:
            for j in st.session_state.features[i]:
                features_merged.append(j)

        st.session_state.feature_df = get_df_from_rules(
            [";".join(feat[0]) for feat in features_merged],
            [";".join(feat[1]) for feat in features_merged],
            [feat[2] for feat in features_merged],
        )

        with st.form("example form") as f:
            gb = GridOptionsBuilder.from_dataframe(st.session_state.feature_df)
            # make all columns editable
            gb.configure_columns(
                ["rules", "negated_rules", "predicted_label"], editable=True
            )
            gb.configure_selection(
                "multiple",
                use_checkbox=True,
                groupSelectsChildren=True,
                groupSelectsFiltered=True,
            )
            go = gb.build()
            ag = AgGrid(
                st.session_state.feature_df,
                gridOptions=go,
                key="grid1",
                allow_unsafe_jscode=True,
                reload_data=True,
                update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.VALUE_CHANGED,
                width="100%",
                theme="light",
                fit_columns_on_grid_load=False,
            )

            delete_or_train = st.radio("Delete selected rules", ("none", "delete"))
            submit = st.form_submit_button(label="save updates")

        if submit:
            delete = delete_or_train == "delete"

            st.session_state.rows_to_delete = [r["rules"] for r in ag["selected_rows"]]
            st.session_state.rls_after_delete = []

            negated_list = ag["data"]["negated_rules"].tolist()
            predicted_labels = ag["data"]["predicted_label"].tolist()
            feature_list = []
            for i, rule in enumerate(ag["data"]["rules"].tolist()):
                if not negated_list[i].strip():
                    feature_list.append([rule.split(";"), [], predicted_labels[i]])
                else:
                    feature_list.append(
                        [
                            rule.split(";"),
                            negated_list[i].strip().split(";"),
                            predicted_labels[i],
                        ]
                    )
            if st.session_state.rows_to_delete and delete:
                for r in feature_list:
                    if ";".join(r[0]) not in st.session_state.rows_to_delete:
                        st.session_state.rls_after_delete.append(r)
            else:
                st.session_state.rls_after_delete = copy.deepcopy(feature_list)

            if st.session_state.rls_after_delete and not delete:
                save_after_modify(hand_made_rules)

        if st.session_state.rows_to_delete and delete_or_train == "delete":
            with st.form("Delete form"):
                st.write("The following rules will be deleted, do you accept it?")
                st.write(st.session_state.rows_to_delete)
                save_button = st.form_submit_button("Accept Delete")

            if save_button:
                save_after_modify(hand_made_rules)
        text = st.text_area("You can add a new rule here manually")
        negated_text = st.text_area("You can modify the negated features here")
        label_text = st.text_area("You can modify the label here")
        agree = st.button("Add rule to the ruleset")
        if agree:
            if not negated_text.strip():
                negated_features = []
            else:
                negated_features = negated_text.split(";")
            if not label_text.strip():
                with st_stdout("error"):
                    print("Please provide label for your rule!")
            else:
                if label_text.strip() not in st.session_state.features:
                    st.session_state.features[label_text.strip()] = []

                st.session_state.features[label_text].append(
                    [[text], negated_features, label_text]
                )
                if st.session_state.features[label_text]:
                    features_merged = []
                    for i in st.session_state.features:
                        for j in st.session_state.features[i]:
                            features_merged.append(j)

                    st.session_state.feature_df = get_df_from_rules(
                        [";".join(feat[0]) for feat in features_merged],
                        [";".join(feat[1]) for feat in features_merged],
                        [feat[2] for feat in features_merged],
                    )
                    save_rules = hand_made_rules or "saved_features.json"
                    save_ruleset(save_rules, st.session_state.features)
                    rerun()

    with col2:
        text_input = st.text_area(
            "Provide the text here you want to parse and apply rules to!"
        )

        if text_input:
            graphs, predicted = match_texts(text_input, extractor, graph_format)
            dot_current_graph = to_dot(
                graphs[0],
            )

            st.markdown("__The predicted labels are:__")
            st.write(predicted[0][0])

            st.markdown("__The matched patterns are:__")
            st.write(predicted[0][1])
            if st.session_state.download:
                graph_pipe = Source(dot_current_graph).pipe(format="svg")
                st.download_button(
                    label="Download graph as SVG",
                    data=graph_pipe,
                    file_name="graph.svg",
                    mime="mage/svg+xml",
                )

            with st.expander("Graph dot source", expanded=False):
                st.write(dot_current_graph)

            st.graphviz_chart(
                dot_current_graph,
                use_container_width=True,
            )

            st.write("Penman format:")
            st.text(penman.encode(penman.decode(graph_to_pn(graphs[0])), indent=10))
            st.write("In one line format:")
            st.write(graph_to_pn(graphs[0]))


def simple_mode(evaluator, data, val_data, graph_format, feature_path, hand_made_rules):
    if hand_made_rules:
        with open(hand_made_rules) as f:
            st.session_state.features = json.load(f)

    if "df" not in st.session_state:
        st.session_state.df = data.copy()
        if "index" not in st.session_state.df:
            # First reset the index, so it starts from 0 and increments sequentially
            # Then reset again and add the index as a column
            st.session_state.df.reset_index(level=0, inplace=True, drop=True)
            st.session_state.df.reset_index(level=0, inplace=True)

    if not feature_path and not st.session_state.trained:
        st.sidebar.title("Train your dataset!")
        show_app = st.sidebar.button("Train")
        st.sidebar.markdown(
            """
        __Min edge in features (if you set it higher than 0, we won't allow single tokens in the features, only edges)__
        """
        )
        st.session_state.min_edge = st.sidebar.number_input(
            "", min_value=0, max_value=3, value=0, step=1
        )
        st.sidebar.markdown(
            """
            __Ranking type:__
             - False -> train a machine learning model and rank based on feature importance
             - True -> rank all features based on TP and FP scores
            """
        )
        st.session_state.rank = st.sidebar.selectbox("", options=[False, True], key=1)

        st.sidebar.markdown(
            """
        __Show download button for graphs (an option to allow downloading graphs)__
        """
        )
        st.session_state.download = st.sidebar.selectbox(
            "", options=[False, True], key=2
        )
        if show_app:
            st.session_state.suggested_features = train_df(
                data, st.session_state.min_edge
            )
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
        st.bar_chart(data.text.str.len().sort_values(ascending=False).iloc[0:100])

        st.write("common words:")
        st.bar_chart(
            pd.Series(" ".join(data["text"]).lower().split()).value_counts()[:100]
        )

    if st.session_state.trained or feature_path:

        with st.expander("Browse dataset:"):
            st.markdown(
                """
                The dataset can be inspected below, if you are interested in of the graphs, you can check it in the graph browser under 'Graphs and results'
                """
            )
            gb = GridOptionsBuilder.from_dataframe(st.session_state.df)
            gb.configure_default_column(
                editable=False,
                resizable=True,
                sorteable=True,
                wrapText=True,
                autoHeight=True,
            )
            gb.configure_column("graph", hide=True)

            go = gb.build()
            selected_df = AgGrid(
                st.session_state.df,
                gridOptions=go,
                allow_unsafe_jscode=True,
                reload_data=False,
                update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.VALUE_CHANGED,
                width="100%",
                theme="light",
                fit_columns_on_grid_load=True,
            )

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
                    else len(st.session_state.suggested_features[key])
                )
                st.session_state.features[key] = [
                    st.session_state.suggested_features[key].pop(0)
                    for _ in range(pop_len)
                ]

        col1.header("Rule chooser and modifier")

        col2.header("Graph viewer and evaluator")

        with col1:
            classes = st.selectbox(
                "First, choose class you want to use to build rules",
                list(st.session_state.features.keys()),
            )

            st.session_state.feature_df = get_df_from_rules(
                [";".join(feat[0]) for feat in st.session_state.features[classes]],
                [";".join(feat[1]) for feat in st.session_state.features[classes]],
            )

            with st.form("example form") as f:

                st.markdown(
                    """
                __You can modify any rule you want to__

                Remember, we use the [PENMAN](https://github.com/goodmami/penman) notation to describe a rule. You can find more information about the rules in the [README](https://github.com/adaamko/POTATO) of our repository.
                """
                )

                gb = GridOptionsBuilder.from_dataframe(st.session_state.feature_df)
                # make all columns editable
                gb.configure_columns(["rules", "negated_rules"], editable=True)
                gb.configure_selection(
                    "multiple",
                    use_checkbox=True,
                    groupSelectsChildren=True,
                    groupSelectsFiltered=True,
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
                    theme="light",
                    fit_columns_on_grid_load=False,
                )

                st.markdown(
                    """
                    After you modified any rule, click on __save updates__ button to save your changes.


                    Tick the __box__ next to the rules you want to evaluate and then, click on the __evaluate__ button.
                    If you don't tick any rule, the evaluator will evaluate all the rules.
                """
                )
                delete_or_train = st.radio(
                    "Delete or Refine selected rules (you can only refine underspecified rules that contain a regexp)",
                    ("none", "delete", "refine"),
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
                train = delete_or_train == "refine"

                st.session_state.rows_to_delete = [
                    r["rules"] for r in ag["selected_rows"]
                ]
                st.session_state.rls_after_delete = []

                feature_list = []
                selected_rules = ag["data"].to_dict(orient="records")

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
                if st.session_state.rows_to_delete and delete:
                    for r in feature_list:
                        if ";".join(r[0]) not in st.session_state.rows_to_delete:
                            st.session_state.rls_after_delete.append(r)
                elif st.session_state.rows_to_delete and train:
                    st.session_state.rls_after_delete = copy.deepcopy(feature_list)
                    rule_to_train = st.session_state.rows_to_delete[0]
                    if ";" in rule_to_train or ".*" not in rule_to_train:
                        st.text("Only single and underspecified rules can be refined!")
                    else:
                        selected_words = evaluator.train_feature(
                            classes, rule_to_train, data, graph_format
                        )

                        for f in selected_words:
                            st.session_state.rls_after_delete.append([[f], [], classes])
                else:
                    st.session_state.rls_after_delete = copy.deepcopy(feature_list)

                if st.session_state.rls_after_delete and not delete:
                    save_after_modify(hand_made_rules, classes)

            if st.session_state.rows_to_delete and delete_or_train == "delete":
                with st.form("Delete form"):
                    st.write("The following rules will be deleted, do you accept it?")
                    st.write(st.session_state.rows_to_delete)
                    save_button = st.form_submit_button("Accept Delete")

                if save_button:
                    save_after_modify(hand_made_rules, classes)

            add_rule_manually(classes, hand_made_rules)
            rank_and_suggest(classes, data, evaluator)

            if st.session_state.ml_feature:
                show_ml_feature(classes, hand_made_rules)

        with col2:
            # THIS IS HERE BECAUSE STREAMLIT IS BUGGY AND DOESN'T DISPLAY TWO DOT GRAPHS IN DIFFERENT CONTAINERS
            # NEXT RELEASE WILL FIX IT
            with st.expander("Browse graphs:"):
                graph_id = st.number_input(
                    label="Input the ID of the graph you want to view", min_value=0
                )

                browse_current_graph_nx = (
                    st.session_state.df[st.session_state.df.index == graph_id]
                    .iloc[0]
                    .graph
                )
                browse_current_graph = to_dot(browse_current_graph_nx)
                if st.session_state.download:
                    graph_pipe = Source(browse_current_graph).pipe(format="svg")
                    st.download_button(
                        label="Download graph as SVG",
                        data=graph_pipe,
                        file_name="graph.svg",
                        mime="mage/svg+xml",
                    )

                st.graphviz_chart(
                    browse_current_graph,
                    use_container_width=True,
                )

                st.write("Penman format:")
                st.text(
                    penman.encode(
                        penman.decode(graph_to_pn(browse_current_graph_nx)), indent=10
                    )
                )
                st.write("In one line format:")
                st.write(graph_to_pn(browse_current_graph_nx))
            if not st.session_state.df_statistics.empty and st.session_state.sens:
                if st.session_state.sens:
                    nodes, option = rule_chooser()
                st.markdown(
                    f"<span>Result of using all the rules: Precision: <b>{st.session_state.whole_accuracy[0]:.3f}</b>, \
                        Recall: <b>{st.session_state.whole_accuracy[1]:.3f}</b>, Fscore: <b>{st.session_state.whole_accuracy[2]:.3f}</b></span>",
                    unsafe_allow_html=True,
                )
                (
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
                ) = extract_data_from_dataframe(option)

                st.markdown(
                    f"<span>The rule's result: Precision: <b>{prec:.3f}</b>, Recall: <b>{recall:.3f}</b>, \
                        Fscore: <b>{fscore:.3f}</b>, True positives: <b>{len(tp_graphs)}</b>, False positives: <b>{len(fp_graphs)}</b></span>",
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
                        graph_viewer("FP", fp_graphs, fp_sentences, fp_indices, nodes)

                elif tp_fp_fn == "True Positive graphs":
                    if tp_graphs:
                        graph_viewer("TP", tp_graphs, tp_sentences, tp_indices, nodes)

                elif tp_fp_fn == "False Negative graphs":
                    if fn_graphs:
                        graph_viewer("FN", fn_graphs, fn_sentences, fn_indices, nodes)


def advanced_mode(evaluator, train_data, graph_format, feature_path, hand_made_rules):
    data = read_train(train_data)
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
        if "index" not in st.session_state.df:
            # First reset the index, so it starts from 0 and increments sequentially
            # Then reset again and add the index as a column
            st.session_state.df.reset_index(level=0, inplace=True, drop=True)
            st.session_state.df.reset_index(level=0, inplace=True)

    df_annotated = st.session_state.df[st.session_state.df.annotated == True][
        ["index", "text", "label", "applied_rules"]
    ]
    df_unannotated = st.session_state.df[st.session_state.df.annotated == False][
        ["index", "text", "label", "applied_rules"]
    ]

    # First we need to provide the labels we want to annotate
    if "labels" not in st.session_state:
        st.text("Before we start, please provide labels you want to train")
        user_input = st.text_input("label encoding", placeholder="NOT:0,OFF:1")

        if st.button("Add labels"):
            if user_input:
                try:
                    labs = user_input.split(",")
                    assert (
                        len(labs) == 2
                    ), "Please provide only two labels, currently we only support binary annotations!"
                    st.session_state.labels = {
                        label.split(":")[0]: int(label.split(":")[1]) for label in labs
                    }

                    st.write(st.session_state.labels)
                    st.session_state.inverse_labels = {
                        v: k for (k, v) in st.session_state.labels.items()
                    }
                    rerun()
                except Exception as e:
                    st.write(e)
                    st.write("Bad format, the right format is NOT:0,OFF:1")
            else:
                st.write("No labels provided!")

    else:
        with st.expander("Annotation/Dataset browser:"):
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
                    autoHeight=True,
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
                    update_mode=GridUpdateMode.MODEL_CHANGED
                    | GridUpdateMode.VALUE_CHANGED,
                    width="100%",
                    theme="light",
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
                    save_dataframe(st.session_state.df, train_data)
                    rerun()

            st.markdown(
                f"<span>Samples you have already annotated:</span>",
                unsafe_allow_html=True,
            )
            with st.form("annotated form") as f:
                gb = GridOptionsBuilder.from_dataframe(df_annotated)
                gb.configure_default_column(
                    editable=True,
                    resizable=True,
                    sorteable=True,
                    wrapText=True,
                )
                # make all columns editable
                gb.configure_selection(
                    "multiple",
                    use_checkbox=True,
                    groupSelectsChildren=True,
                    groupSelectsFiltered=True,
                )
                go = gb.build()
                ag_ann = AgGrid(
                    df_annotated,
                    gridOptions=go,
                    key="grid3",
                    allow_unsafe_jscode=True,
                    reload_data=True,
                    update_mode=GridUpdateMode.MODEL_CHANGED
                    | GridUpdateMode.VALUE_CHANGED,
                    width="100%",
                    theme="light",
                    fit_columns_on_grid_load=True,
                )

                clear_annotate = st.form_submit_button("Clear annotation")

            if clear_annotate:
                if ag_ann["selected_rows"]:
                    for row in ag_ann["selected_rows"]:
                        st.session_state.df.loc[
                            row["index"], "label"
                        ] = st.session_state.inverse_labels[1]
                        st.session_state.df.loc[row["index"], "annotated"] = False
                        st.session_state.df.loc[row["index"], "label"] = ""
                    save_dataframe(st.session_state.df, train_data)
                    rerun()

        st.sidebar.title("Train your dataset!")
        train = st.sidebar.button("Train!")
        st.sidebar.markdown(
            """
        __Min edge in features (if you set it higher than 0, we won't allow single tokens in the features, only edges)__
        """
        )
        st.session_state.min_edge = st.sidebar.number_input(
            "", min_value=0, max_value=3, value=0, step=1
        )
        st.sidebar.markdown(
            """
            __Ranking type:__
             - False -> train a machine learning model and rank based on feature importance
             - True -> rank all features based on TP and FP scores
            """
        )
        st.session_state.rank = st.sidebar.selectbox("", options=[False, True], key=1)

        st.sidebar.markdown(
            """
        __Show download button for graphs (an option to allow downloading graphs)__
        """
        )
        st.session_state.download = st.sidebar.selectbox(
            "", options=[False, True], key=2
        )
        if train:
            df_to_train = st.session_state.df.copy()
            # df_to_train = df_to_train[df_to_train.applied_rules.map(len) == 0]

            if not df_to_train.empty:
                st.session_state.trained = True
                df_to_train["label"] = df_to_train["label"].apply(
                    lambda x: st.session_state.inverse_labels[0] if not x else x
                )
                df_to_train["label_id"] = df_to_train["label"].apply(
                    lambda x: st.session_state.labels[x]
                )

                positive_size = df_to_train.groupby("label").size()[
                    st.session_state.inverse_labels[1]
                ]
                df_to_train = df_to_train.groupby("label").sample(
                    n=positive_size, random_state=1, replace=True
                )
                st.session_state.suggested_features = train_df(
                    df_to_train, st.session_state.min_edge, st.session_state.rank
                )
                st.session_state.df_to_train = df_to_train
                st.session_state.df_statistics = pd.DataFrame

            else:
                st.write("Empty dataframe!")

        col1, col2 = st.columns(2)

        with col1:
            if not st.session_state.features:
                st.session_state.features[st.session_state.inverse_labels[1]] = []

            classes = st.selectbox(
                "First, choose class you want to use to build rules",
                list(st.session_state.features.keys()),
            )

            if not st.session_state.features[classes]:
                st.session_state.features[classes] = [
                    [["(u_1 / dummy :1 (u_2 / feature))"], [], classes]
                ]
            st.session_state.feature_df = get_df_from_rules(
                [";".join(feat[0]) for feat in st.session_state.features[classes]],
                [";".join(feat[1]) for feat in st.session_state.features[classes]],
            )

            with st.form("example form") as f:
                st.markdown(
                    """
                __You can modify any rule you want to__

                Remember, we use the [PENMAN](https://github.com/goodmami/penman) notation to describe a rule. You can find more information about the rules in the [README](https://github.com/adaamko/POTATO) of our repository.
                """
                )

                gb = GridOptionsBuilder.from_dataframe(st.session_state.feature_df)
                # make all columns editable
                gb.configure_columns(["rules", "negated_rules"], editable=True)
                gb.configure_selection(
                    "multiple",
                    use_checkbox=True,
                    groupSelectsChildren=True,
                    groupSelectsFiltered=True,
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
                    theme="light",
                    fit_columns_on_grid_load=True,
                )

                st.markdown(
                    """
                    After you modified any rule, click on __save updates__ button to save your changes.


                    Tick the __box__ next to the rules you want to evaluate and then, click on the __evaluate__ button.
                    If you don't tick any rule, the evaluator will evaluate all the rules.

                    If you think your rules are good and you want to evaluate samples based on them, click on the __annotate__ button.
                """
                )

                delete_or_train = st.radio(
                    "Delete or Refine selected rules (you can only refine underspecified rules that contain a regexp)",
                    ("none", "delete", "refine"),
                )
                submit = st.form_submit_button(label="save updates")
                evaluate = st.form_submit_button(label="evaluate selected")
                annotate = st.form_submit_button(label="annotate based on selected")

            if evaluate or annotate:
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
                if not st.session_state.sens:
                    with st_stdout("error"):
                        print(
                            "There are no rules to evaluate, please write some rules first, or get suggestions by our system!"
                        )

                else:
                    with st.spinner("Evaluating rules..."):
                        (
                            st.session_state.df_statistics,
                            st.session_state.whole_accuracy,
                        ) = evaluator.evaluate_feature(
                            classes,
                            feature_list,
                            st.session_state.df,
                            graph_format,
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
                train = delete_or_train == "refine"

                st.session_state.rows_to_delete = [
                    r["rules"] for r in ag["selected_rows"]
                ]
                st.session_state.rls_after_delete = []

                feature_list = []
                selected_rules = ag["data"].to_dict(orient="records")

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
                if st.session_state.rows_to_delete and delete:
                    for r in feature_list:
                        if ";".join(r[0]) not in st.session_state.rows_to_delete:
                            st.session_state.rls_after_delete.append(r)
                elif st.session_state.rows_to_delete and train:
                    st.session_state.rls_after_delete = copy.deepcopy(feature_list)
                    rule_to_train = st.session_state.rows_to_delete[0]
                    if ";" in rule_to_train or ".*" not in rule_to_train:
                        st.text("Only single and underspecified rules can be refined!")
                    else:
                        selected_words = evaluator.train_feature(
                            classes,
                            rule_to_train,
                            st.session_state.df,
                            graph_format,
                        )

                        for f in selected_words:
                            st.session_state.rls_after_delete.append([[f], [], classes])
                else:
                    st.session_state.rls_after_delete = copy.deepcopy(feature_list)

                if st.session_state.rls_after_delete and not delete:
                    save_after_modify(hand_made_rules, classes)

            if st.session_state.rows_to_delete and delete_or_train == "delete":
                with st.form("Delete form"):
                    st.write("The following rules will be deleted, do you accept it?")
                    st.write(st.session_state.rows_to_delete)
                    save_button = st.form_submit_button("Accept Delete")

                if save_button:
                    save_after_modify(hand_made_rules, classes)

            add_rule_manually(classes, hand_made_rules)
            rank_and_suggest(classes, st.session_state.df, evaluator)

            if st.session_state.ml_feature:
                show_ml_feature(classes, hand_made_rules)
        with col2:
            with st.expander("Browse dataset/graphs"):
                graph_id = st.number_input(
                    label="Input the ID of the graph you want to view", min_value=0
                )

                browse_current_graph_nx = (
                    st.session_state.df[st.session_state.df.index == graph_id]
                    .iloc[0]
                    .graph
                )
                browse_current_graph = to_dot(browse_current_graph_nx)
                if st.session_state.download:
                    graph_pipe = Source(browse_current_graph).pipe(format="svg")
                    st.download_button(
                        label="Download graph as SVG",
                        data=graph_pipe,
                        file_name="graph.svg",
                        mime="mage/svg+xml",
                    )

                st.graphviz_chart(
                    browse_current_graph,
                    use_container_width=True,
                )
                st.write("Penman format:")
                st.text(
                    penman.encode(
                        penman.decode(graph_to_pn(browse_current_graph_nx)),
                        indent=10,
                    )
                )
                st.write("In one line format:")
                st.write(graph_to_pn(browse_current_graph_nx))

            if not st.session_state.df_statistics.empty and st.session_state.sens:
                if st.session_state.sens:
                    nodes, option = rule_chooser()

                st.markdown(
                    f"<span>Result of using all the rules: Precision: <b>{st.session_state.whole_accuracy[0]:.3f}</b>, \
                        Recall: <b>{st.session_state.whole_accuracy[1]:.3f}</b>, Fscore: <b>{st.session_state.whole_accuracy[2]:.3f}</b></span>",
                    unsafe_allow_html=True,
                )

                (
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
                ) = extract_data_from_dataframe(option)

                st.markdown(
                    f"<span>The rule's result: Precision: <b>{prec:.3f}</b>, Recall: <b>{recall:.3f}</b>, \
                        Fscore: <b>{fscore:.3f}</b>, True positives: <b>{len(tp_graphs)}</b>, False positives: <b>{len(fp_graphs)}</b></span>",
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

                if tp_fp_fn == "Predicted":
                    predicted_inds = [
                        i for i, pred in enumerate(predicted) if pred == 1
                    ]
                    filt_df = st.session_state.df[
                        st.session_state.df.index.isin(predicted_inds)
                    ]
                    pred_graphs = filt_df.graph.tolist()
                    pred_sentences = [
                        (sen[0], sen[1])
                        for sen in zip(filt_df.text.tolist(), filt_df.label.tolist())
                    ]
                    graph_viewer(
                        "PD", pred_graphs, pred_sentences, predicted_inds, nodes
                    )

                elif tp_fp_fn == "False Positive graphs":
                    if fp_graphs:
                        graph_viewer("FP", fp_graphs, fp_sentences, fp_indices, nodes)

                elif tp_fp_fn == "True Positive graphs":
                    if tp_graphs:
                        graph_viewer("TP", tp_graphs, tp_sentences, tp_indices, nodes)

                elif tp_fp_fn == "False Negative graphs":
                    if fn_graphs:
                        graph_viewer("FN", fn_graphs, fn_sentences, fn_indices, nodes)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--train-data", type=str)
    parser.add_argument("-v", "--val-data", type=str)
    parser.add_argument(
        "-sr",
        "--suggested-rules",
        default=None,
        type=str,
        help="Rules extracted automatically from python. If not present, the UI will automatically train it.",
    )
    parser.add_argument(
        "-hr",
        "--hand-rules",
        default=None,
        type=str,
        help="Rules extracted with the UI. If provided, the UI will load them.",
    )
    parser.add_argument("-m", "--mode", default="simple", type=str)
    parser.add_argument("-g", "--graph-format", default="fourlang", type=str)
    parser.add_argument(
        "-l",
        "--label",
        default=None,
        type=str,
        help="Specify label for OneVsAll multi-label classification. Datasets require a labels column with all valid labels.",
    )
    return parser.parse_args()


def main(args):
    st.set_page_config(layout="wide")
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {
	
                visibility: hidden;
                
                }
            footer:after {
                content:'POTATO'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
            </style>
            
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center; color: black;'>POTATO</h1>"
        "<h2 style='text-align: center; color: black;'>explainable information extraction framework</h2>"
        "<h3 style='text-align: center;'>Visit our <a href='https://github.com/adaamko/POTATO'>GitHub</a> page for more information and to access the code</h3><p></p>",
        unsafe_allow_html=True,
    )

    init_session_states()
    evaluator = init_evaluator()
    if args.train_data:
        data = read_train(args.train_data, args.label)
    if args.val_data:
        val_data = read_val(args.val_data, args.label)
    graph_format = args.graph_format
    feature_path = args.suggested_rules
    hand_made_rules = args.hand_rules
    mode = args.mode
    if mode == "simple":
        assert args.train_data, "Train data is required for simple mode"
        assert args.val_data, "Validation data is required for simple mode"
        simple_mode(
            evaluator, data, val_data, graph_format, feature_path, hand_made_rules
        )
    elif mode == "advanced":
        assert args.train_data, "Train data is required for advanced mode"
        advanced_mode(
            evaluator, args.train_data, graph_format, feature_path, hand_made_rules
        )
    # For inference mode you don't need any training data as we will use the model
    # to predict the labels on-the-fly
    elif mode == "inference":
        inference_mode(evaluator, hand_made_rules)

    else:
        st.error("Mode not supported")


if __name__ == "__main__":
    args = get_args()
    main(args)
