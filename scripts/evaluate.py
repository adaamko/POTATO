import argparse
import json
import logging
import os
import sys
from ast import literal_eval

import pandas as pd

from tuw_nlp.common.eval import get_cat_stats, print_cat_stats
from tuw_nlp.graph.utils import graph_to_pn
from xpotato.graph_extractor.extract import FeatureEvaluator
from xpotato.graph_extractor.graph import PotatoGraph
from xpotato.graph_extractor.rule import RuleSet


# TODO Adam: This is not the best place for these functions but I didn't want it to be in the frontend.utils
# ------------------------------------------------------


def filter_label(df, labels):
    df["label"] = df.apply(
        lambda x: x["label"] if x["label"] in labels else "NOT", axis=1
    )
    df["label_id"] = df.apply(lambda x: 0 if x["label"] == "NOT" else 1, axis=1)
    if "labels" in df:
        df["labels"] = df.apply(
            lambda x: [label for label in x["labels"] if label in labels], axis=1
        )


def read_df(path, labels=None, binary=False):
    if binary:
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path, sep="\t", converters={"labels": literal_eval})
        graphs = []
        for graph in df["graph"]:
            potato_graph = PotatoGraph(graph_str=graph)
            graphs.append(potato_graph.graph)
        df["graph"] = graphs
    if labels is not None:
        filter_label(df, labels)
    return df


def get_features(path, label=None):
    files = []
    if os.path.isfile(path):
        assert path.endswith("json") or path.endswith(
            "tsv"
        ), "features file must be JSON or TSV"
        files.append(path)
    elif os.path.isdir(path):
        for fn in os.listdir(path):
            assert fn.endswith("json") or fn.endswith(
                "tsv"
            ), f"feature dir should only contain JSON or TSV files: {fn}"
            files.append(os.path.join(path, fn))
    else:
        raise ValueError(f"not a file or directory: {path}")

    labels = set()
    all_features = []
    for fn in files:
        ruleset = RuleSet()
        if fn.endswith("json"):
            ruleset.from_json(fn)
        elif fn.endswith("tsv"):
            ruleset.from_tsv(fn)
        else:
            raise ValueError(f"unknown file type: {fn}")
        features = ruleset.to_list()
        for k in features:
            lab = k[2]
            if label and label != lab:
                continue
            labels.add(lab)
            all_features.append(k)

    return all_features, labels


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--graph-type", type=str, default="fourlang")
    parser.add_argument("-f", "--features", type=str, required=True)
    parser.add_argument("-d", "--dataset-path", type=str, default=None, required=True)
    parser.add_argument("-m", "--mode", type=str, default="predictions")
    parser.add_argument(
        "-l",
        "--label",
        default=None,
        type=str,
        help="Specify label for OneVsAll multi-label classification. Datasets require a labels column with all valid labels.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s : "
        + "%(module)s (%(lineno)s) - %(levelname)s - %(message)s",
    )
    args = get_args()
    assert args.mode in ("predictions", "report")

    features, labels = get_features(args.features, args.label)

    df = read_df(args.dataset_path, labels)

    evaluator = FeatureEvaluator()

    pred_df = evaluator.match_features(df, features, multi=True)

    if args.mode == "predictions":
        pred_df.to_csv(sys.stdout, sep="\t")
    else:
        if "labels" in df:
            gold_labels = df.labels.tolist()
        elif "label" in df and df["label"].iloc[0]:
            gold_labels = [[label] for label in df.label]
        else:
            raise ValueError(
                "There are no labels in the dataset, we cannot generate a classification report. Are you evaluating a test set?"
            )

        print_cat_stats(get_cat_stats(pred_df["Predicted label"], gold_labels))


if __name__ == "__main__":
    main()
