import json
import os
import numpy as np
import logging
from typing import List, Dict
from argparse import ArgumentParser, ArgumentError

from sklearn.model_selection import train_test_split
from xpotato.dataset.explainable_dataset import ExplainableDataset
from xpotato.models.trainer import GraphTrainer
from xpotato.dataset.utils import save_dataframe


def read_json(file_path: str) -> Dict[str, List[Dict[str, List[str]]]]:
    data_by_target = {}
    with open(file_path) as dataset:
        data = json.load(dataset)
        for post in data.values():
            sentence = " ".join(post["post_tokens"])
            targets = {}
            target = []
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
                    if target[0] not in data_by_target:
                        data_by_target[target[0]] = []
                    data_by_target[target[0]].append(
                        {
                            "tokens": post["post_tokens"],
                            "sentence": sentence,
                            "rationale": rationale,
                        }
                    )
    return data_by_target


def process(data_path: str, groups: List[str], target: str, just_none: bool):
    running_groups = ["none", target] if just_none else groups
    sentences = []
    for group in running_groups:
        group_path = os.path.join(data_path, f"{group}.json")
        if os.path.isfile(group_path):
            with open(group_path, "r") as group_json:
                group_list = json.load(group_json)
                sentences += [
                    (
                        example["sentence"],
                        "None" if group != target else target.capitalize(),
                        [
                            tok
                            for (rat, tok) in zip(
                                example["rationale"], example["tokens"]
                            )
                            if rat == 1
                        ]
                        if group == target
                        else [],
                    )
                    for example in group_list
                ]
        else:
            logging.warning(f"Skipping {group}, because {group_path} does not exist.")

    potato_dataset = ExplainableDataset(
        sentences, label_vocab={"None": 0, f"{target.capitalize()}": 1}, lang="en"
    )
    potato_dataset.set_graphs(potato_dataset.parse_graphs(graph_format="ud"))
    df = potato_dataset.to_dataframe()
    """
    trainer = GraphTrainer(df)
    features = trainer.prepare_and_train()
    """
    train, val = train_test_split(df, test_size=0.2, random_state=1234)
    save_dataframe(train, os.path.join(data_path, "train.tsv"))
    save_dataframe(val, os.path.join(data_path, "val.tsv"))

    """
    with open("features.json", "w+") as f:
        json.dump(features, f)
    """


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
        action="store_true",
        help="Use only the normal texts as counter.",
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
        if not os.path.isfile(dataset):
            raise ArgumentError(
                "The specified data path is not a file and does not contain a dataset.json file. "
                "If your file has a different name, please specify."
            )
        dir_path = os.path.dirname(dataset)
        dt_by_target = read_json(dataset)
        for name, list_of_dicts in dt_by_target.items():
            with open(os.path.join(dir_path, f"{name.lower()}.json"), "w") as json_file:
                json.dump(list_of_dicts, json_file, indent=4)

        if args.mode == "both":
            process(
                data_path=dir_path,
                groups=target_groups,
                target=args.target,
                just_none=args.just_none,
            )

    else:
        dir_path = (
            os.path.dirname(args.data_path)
            if os.path.isfile(args.data_path)
            else args.data_path
        )
        process(
            data_path=dir_path,
            groups=target_groups,
            target=args.target,
            just_none=args.just_none,
        )
