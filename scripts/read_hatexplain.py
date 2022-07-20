import json
import os
import numpy as np
import logging
from typing import List, Dict, Tuple
from argparse import ArgumentParser, ArgumentError
from ast import literal_eval

import pandas as pd
from sklearn.model_selection import train_test_split
from tuw_nlp.text.preprocess.hatexplain import preprocess_hatexplain
from xpotato.dataset.explainable_dataset import ExplainableDataset
from xpotato.models.trainer import GraphTrainer
from xpotato.dataset.utils import save_dataframe


def read_json(
    file_path: str, min_token_count: int = 2
) -> List[Dict[str, List[Dict[str, List[str]]]]]:
    data_by_target = []
    with open(file_path) as dataset:
        data = json.load(dataset)
        for post in data.values():
            sentence = " ".join(post["post_tokens"])
            sentence = preprocess_hatexplain(sentence)
            targets = {}
            labels = {}
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
            if len(labels) != len(post["annotators"]):
                label = max(labels.items(), key=lambda x: x[1])[0]
                if label == "normal":
                    target = ["None"]
                else:
                    target = [t[0] for t in targets.items() if t[1] > 1]
                rationale = []
                if len(post["rationales"]) > 0:
                    rats = [
                        n
                        for n in post["rationales"]
                        if len(n) == len(post["post_tokens"])
                    ]
                    rationale = np.round(np.mean(rats, axis=0), decimals=0).tolist()
                if len(target) == 1:
                    other_targets = []
                    if min_token_count < 2:
                        other_targets = [t for t in targets.keys() if t != target[0]]
                    data_by_target.append(
                        {
                            "id": post["post_id"],
                            "tokens": post["post_tokens"],
                            "sentence": sentence,
                            "rationale": rationale,
                            "label": target[0],
                            "secondary_labels": other_targets,
                        }
                    )
    return data_by_target


def get_sentences(
    group: pd.DataFrame, other: pd.DataFrame, target: str
) -> List[Tuple[str, str, List[str]]]:
    sentences = {
        index: (
            example.sentence,
            target.capitalize(),
            [
                tokens
                for (rationale, tokens) in zip(
                    literal_eval(example.rationale), literal_eval(example.tokens)
                )
                if rationale == 1
            ],
        )
        for index, example in group.iterrows()
    }
    sentences.update(
        {index: (example.sentence, "None", []) for index, example in other.iterrows()}
    )
    return [s[1] for s in sorted(sentences.items())]


def process(
    data_path: str,
    target: str,
    just_none: bool,
    split_file: str,
    use_secondary: bool = False,
    create_features: bool = False,
) -> None:
    df = pd.read_csv(os.path.join(data_path, "dataset.tsv"), sep="\t")
    split_ids = json.load(open(split_file))
    train_df = df[df.id.isin(split_ids["train"])]
    val_df = df[df.id.isin(split_ids["val"])]
    test_df = df[df.id.isin(split_ids["test"])]
    feature_trainer_df = None

    for dataframe, name in zip((train_df, val_df, test_df), ("train", "val", "test")):
        main_group = dataframe[dataframe.label == target.capitalize()]
        main_others = (
            dataframe[dataframe.label != target.capitalize()]
            if not just_none
            else dataframe[dataframe.label == "None"]
        )
        main_sentences = get_sentences(main_group, main_others, target)

        main_potato_dataset = ExplainableDataset(
            main_sentences,
            label_vocab={"None": 0, f"{target.capitalize()}": 1},
            lang="en",
        )
        graphs = main_potato_dataset.parse_graphs(graph_format="ud")
        main_potato_dataset.set_graphs(graphs)
        main_df = main_potato_dataset.to_dataframe()
        save_dataframe(main_df, os.path.join(data_path, f"{name}.tsv"))

        if use_secondary:
            secondary_group = dataframe[
                (dataframe.secondary_labels.str.contains(target.capitalize()))
                | (dataframe.label == target.capitalize())
            ]
            secondary_other = dataframe.loc[
                set(dataframe.index.tolist()).difference(secondary_group.index.tolist())
            ]
            secondary_sentences = get_sentences(
                secondary_group, secondary_other, target
            )
            secondary_potato_dataset = ExplainableDataset(
                secondary_sentences,
                label_vocab={"None": 0, f"{target.capitalize()}": 1},
                lang="en",
            )
            secondary_potato_dataset.set_graphs(graphs)
            secondary_df = secondary_potato_dataset.to_dataframe()
            save_dataframe(
                secondary_df, os.path.join(data_path, f"secondary_{name}.tsv")
            )
        if feature_trainer_df is None:
            feature_trainer_df = main_df

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
        "--just_none",
        "-n",
        help="Use only the normal texts as counter.",
        action="store_true",
    )
    argparser.add_argument(
        "--min_target",
        "-mt",
        help="The minimum number of annotators finding the text to target a group.",
        type=int,
        default=2,
    )
    argparser.add_argument(
        "--create_features",
        "-cf",
        help="Whether to create train features based on the POTATO graph.",
        action="store_true",
    )
    args = argparser.parse_args()

    if args.mode != "distinct" and args.target is None:
        raise ArgumentError(None,
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
            raise ArgumentError(None,
                "The specified data path is not a file and does not contain a dataset.json file. "
                "If your file has a different name, please specify."
            )
        dir_path = os.path.dirname(dataset)
        dt_by_target = read_json(dataset, args.min_target)
        dataf = pd.DataFrame.from_records(dt_by_target)
        dataf.to_csv(os.path.join(dir_path, "dataset.tsv"), sep="\t")

        if args.mode == "both":
            process(
                data_path=dir_path,
                target=args.target,
                just_none=args.just_none,
                split_file=split,
                use_secondary=args.min_target != 2,
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
            just_none=args.just_none,
            split_file=split,
            use_secondary=args.min_target != 2,
            create_features=args.create_features,
        )
