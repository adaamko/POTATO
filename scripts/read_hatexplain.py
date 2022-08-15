import json
import os
import numpy as np
import logging
from collections import defaultdict
from typing import List, Dict, Tuple
from argparse import ArgumentParser, ArgumentError
from ast import literal_eval

import pandas as pd
from networkx.readwrite import json_graph
from tuw_nlp.text.preprocess.hatexplain import preprocess_hatexplain
from xpotato.dataset.explainable_dataset import ExplainableDataset
from xpotato.graph_extractor.extract import GraphExtractor
from xpotato.graph_extractor.graph import PotatoGraph
from xpotato.models.trainer import GraphTrainer
from xpotato.dataset.utils import save_dataframe


def read_json(
    file_path: str, parse_graphs: bool = True
) -> List[Dict[str, List[Dict[str, List[str]]]]]:
    data_by_target = []
    with open(file_path) as dataset:
        data = json.load(dataset)
        for post in data.values():
            sentence = " ".join(post["post_tokens"])
            sentence = preprocess_hatexplain(sentence)
            targets = {}
            labels = {}
            pure = False
            one_majority = False
            for annotation in post["annotators"]:
                if annotation["label"] not in labels:
                    labels[annotation["label"]] = 1
                else:
                    labels[annotation["label"]] += 1
                for target_i in annotation["target"]:
                    if target_i not in targets:
                        targets[target_i] = 1
                    else:
                        targets[target_i] += 1
            # Pure is if it targets only one (or none) groups
            if len(targets) == 1 or (len(targets) == 2 and "None" in targets):
                pure = True
            # One majority is if the majority vote would just be one target
            if len([l for l in targets.values() if l > 1]) == 1:
                one_majority = True

            # We don't care about the instances, where each annotator said something different label-wise
            if len(labels) != len(post["annotators"]):
                majority_targets = [t for (t, c) in targets.items() if c >= 2]
                minority_targets = [t for (t, c) in targets.items() if c < 2]
                rationale = defaultdict(list)
                # Get the rationales in an organized manner
                if len(post["rationales"]) > 0:
                    not_none_annotators = [
                        a["target"]
                        for a in post["annotators"]
                        if a["label"] != "normal"
                    ]
                    for annotator, rationales in zip(
                        not_none_annotators, post["rationales"]
                    ):
                        major_intersection = list(
                            set(annotator).intersection(majority_targets)
                        )
                        minor_intersection = list(
                            set(annotator).intersection(minority_targets)
                        )
                        for mi in major_intersection + minor_intersection:
                            rationale[mi].append(rationales)
                    rationale = {
                        key: np.round(np.mean(value, axis=0), decimals=0).tolist()
                        for (key, value) in rationale.items()
                    }
                data_by_target.append(
                    {
                        "id": post["post_id"],
                        "tokens": post["post_tokens"],
                        "sentence": sentence,
                        "pure": pure,
                        "one_majority": one_majority,
                        "rationales": dict(rationale),
                        "majority_labels": majority_targets,
                        "minority_labels": minority_targets,
                    }
                )
    if parse_graphs:
        extractor = GraphExtractor(lang="en")
        graphs = list(
            extractor.parse_iterable(
                [data_point["sentence"] for data_point in data_by_target], "ud"
            )
        )
        for graph, data_point in zip(graphs, data_by_target):
            data_point["graph"] = json_graph.adjacency_data(graph)
    return data_by_target


def get_sentences(
    group: pd.DataFrame, other: pd.DataFrame, target: str
) -> List[Tuple[str, str, List[str]]]:
    sentences = {
        index: (
            example.sentence,
            target.capitalize(),
            []
            if target.capitalize() not in literal_eval(example.rationales)
            else [
                tok
                for rat, tok in zip(
                    literal_eval(example.rationales)[target.capitalize()],
                    literal_eval(example.tokens),
                )
                if rat == 1
            ],
            []
            if target.capitalize() not in literal_eval(example.rationales)
            else [
                index
                for index, rat in enumerate(
                    literal_eval(example.rationales)[target.capitalize()]
                )
                if rat == 1
            ],
            []
            if target.capitalize() not in literal_eval(example.rationales)
            else [
                tok["name"]
                for rat, tok in zip(
                    literal_eval(example.rationales)[target.capitalize()],
                    # LT and GT appear only around user or censored as well as an emoji,
                    # but that will not influence this negatively
                    sorted([node for node in literal_eval(example.graph)["nodes"] if node["name"] not in ["LT", "GT"]], key=lambda x: x["id"])[1:],
                )
                if rat == 1
            ],
            PotatoGraph(graph=json_graph.adjacency_graph(literal_eval(example.graph))),
        )
        for (index, example) in group.iterrows()
    }
    sentences.update(
        {
            index: (
                example.sentence,
                "None",
                [],
                [],
                [],
                PotatoGraph(
                    graph=json_graph.adjacency_graph(literal_eval(example.graph))
                ),
            )
            for index, example in other.iterrows()
        }
    )
    return [s[1] for s in sorted(sentences.items())]


def convert_to_potato(group, other, target):
    sentences = get_sentences(group, other, target)
    potato_dataset = ExplainableDataset(
        sentences,
        label_vocab={"None": 0, f"{target.capitalize()}": 1},
        lang="en",
    )
    return potato_dataset.to_dataframe()


def process(
    data_path: str,
    target: str,
    split_file: str,
    create_features: bool = False,
) -> None:
    df = pd.read_csv(os.path.join(data_path, "dataset_02.tsv"), sep="\t")

    save_path = os.path.join(data_path, target)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    split_ids = json.load(open(split_file))
    train_df = df[df.id.isin(split_ids["train"])]
    val_df = df[df.id.isin(split_ids["val"])]
    test_df = df[df.id.isin(split_ids["test"])]
    feature_trainer_df = None

    for dataframe, name in zip((train_df, val_df, test_df), ("train", "val", "test")):
        purity_filters = {
            "pure": dataframe.pure,
            "one_majority": dataframe.one_majority,
            "all": True,
        }
        for purity, purity_filter in purity_filters.items():
            majority_group = dataframe[
                dataframe.majority_labels.apply(lambda x: target.capitalize() in x)
                & purity_filter
            ]
            majority_others = dataframe[
                dataframe.majority_labels.apply(lambda x: target.capitalize() not in x)
                & purity_filter
            ]
            majority_df = convert_to_potato(majority_group, majority_others, target)
            save_dataframe(
                majority_df, os.path.join(save_path, f"majority_{name}_{purity}.tsv")
            )

            minority_group = dataframe[
                (
                    (
                        dataframe.minority_labels.apply(
                            lambda x: target.capitalize() in x
                        )
                    )
                    | (
                        dataframe.majority_labels.apply(
                            lambda x: target.capitalize() in x
                        )
                    )
                )
                & purity_filter
            ]
            minority_other = dataframe[
                (
                    (
                        dataframe.minority_labels.apply(
                            lambda x: target.capitalize() not in x
                        )
                    )
                    & (
                        dataframe.majority_labels.apply(
                            lambda x: target.capitalize() not in x
                        )
                    )
                )
                & purity_filter
            ]
            minority_df = convert_to_potato(minority_group, minority_other, target)
            save_dataframe(
                minority_df, os.path.join(save_path, f"minority_{name}_{purity}.tsv")
            )
        if feature_trainer_df is None:
            feature_trainer_df = majority_df

    if create_features:
        trainer = GraphTrainer(feature_trainer_df)
        features = trainer.prepare_and_train()

        with open(os.path.join(data_path, "features.json"), "w+") as f:
            json.dump(features, f)


if __name__ == "__main__":
    target_groups = [
        "african",
        "arab",
        "asian",
        "caucasian",
        "christian",
        "disability",
        "economic",
        "hindu",
        "hispanic",
        "homosexual",
        "indian",
        "islam",
        "jewish",
        "men",
        "other",
        "refugee",
        "women",
    ]
    argparser = ArgumentParser()
    argparser.add_argument(
        "--data_path", "-d", help="Path to the json dataset.", required=True
    )
    argparser.add_argument("--split_path", "-s", help="Path of the official split.")
    argparser.add_argument(
        "--mode",
        "-m",
        help="Mode to start the program. Modes:"
        "\n\t- distinct: "
        "cut the dataset.json into distinct categorical json files"
        "\n\t- process: "
        "load the chosen category as the target and every other one as non-target"
        "\n\t- both: "
        "run the distinct and the process after eachother",
        default="both",
        choices=["distinct", "process", "both"],
    )
    argparser.add_argument(
        "--target",
        "-t",
        help="The target group to set as our category.",
        choices=target_groups,
    )
    argparser.add_argument(
        "--create_features",
        "-cf",
        help="Whether to create train features based on the POTATO graph.",
        action="store_true",
    )
    args = argparser.parse_args()

    if args.mode != "distinct" and args.target is None:
        raise ArgumentError(
            "Target is not given! If you want to produce a POTATO dataset "
            "(by running this code in process or both mode), you should specify the target."
        )

    if args.mode != "process":
        dataset = (
            args.data_path
            if os.path.isfile(args.data_path)
            else os.path.join(args.data_path, "dataset.json")
        )
        if args.split_path is None:
            args.split_path = args.data_path
        split = (
            args.split_path
            if os.path.isfile(args.split_path)
            else os.path.join(args.split_path, "post_id_divisions.json")
        )
        if not os.path.isfile(dataset):
            raise ArgumentError(
                "The specified data path is not a file and does not contain a dataset.json file. "
                "If your file has a different name, please specify."
            )
        dir_path = os.path.dirname(dataset)
        dt_by_target = read_json(dataset)
        dataf = pd.DataFrame.from_records(dt_by_target)
        dataf.to_csv(os.path.join(dir_path, "dataset_02.tsv"), sep="\t")

        if args.mode == "both":
            process(
                data_path=dir_path,
                target=args.target,
                split_file=split,
                create_features=args.create_features,
            )

    else:
        dir_path = (
            os.path.dirname(args.data_path)
            if os.path.isfile(args.data_path)
            else args.data_path
        )
        if args.split_path is None:
            args.split_path = dir_path
        split = (
            args.split_path
            if os.path.isfile(args.split_path)
            else os.path.join(args.split_path, "post_id_divisions.json")
        )
        process(
            data_path=dir_path,
            target=args.target,
            split_file=split,
            create_features=args.create_features,
        )
