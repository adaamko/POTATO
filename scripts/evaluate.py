import argparse
import json
import logging
import sys

import pandas as pd
from sklearn.metrics import classification_report

from xpotato.graph_extractor.extract import FeatureEvaluator
from xpotato.graph_extractor.graph import PotatoGraph


# TODO Adam: This is not the best place for these functions but I didn't want it to be in the frontend.utils
# ------------------------------------------------------


def filter_label(df, label):
    df["label"] = df.apply(lambda x: label if label in x["labels"] else "NOT", axis=1)
    df["label_id"] = df.apply(lambda x: 0 if x["label"] == "NOT" else 1, axis=1)


def read_df(path, label=None, binary=False):
    if binary:
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path, sep="\t")
        graphs = []
        for graph in df["graph"]:
            potato_graph = PotatoGraph(graph_str=graph)
            graphs.append(potato_graph.graph)
        df["graph"] = graphs
    if label is not None:
        filter_label(df, label)
    return df


# ------------------------------------------------------


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
    df = read_df(args.dataset_path, args.label)

    with open(args.features) as f:
        features = json.load(f)

    feature_values = []
    for k in features:
        for f in features[k]:
            feature_values.append(f)
    evaluator = FeatureEvaluator()
    pred_df = evaluator.match_features(df, feature_values)
    report = None
    if "label" in df and df["label"].iloc[0]:
        pred_df["label"] = df.label

        label_to_id = {}

        for label in df.groupby("label").size().keys().tolist():
            if label in label_to_id:
                continue
            else:
                label_to_id[label] = df[df.label == label].iloc[1].label_id

        predicted_label = []
        gold = df.label_id.tolist()

        for label in pred_df["Predicted label"]:
            if label in label_to_id:
                predicted_label.append(label_to_id[label])
            else:
                predicted_label.append(0)

        report = classification_report(gold, predicted_label, digits=3)

    if args.mode == "predictions":
        pred_df.to_csv(sys.stdout, sep="\t")
    elif args.mode == "report":
        assert (
            report
        ), "There are no labels in the dataset, we cannot generate a classification report. Are you evaluating a test set?"
        print(report)


if __name__ == "__main__":
    main()
