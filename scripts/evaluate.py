import argparse
import logging
import pickle
import sys
import json
from collections import defaultdict

import networkx as nx
import pandas as pd
from sklearn.metrics import classification_report
from xpotato.dataset.utils import default_pn_to_graph
from xpotato.graph_extractor.extract import FeatureEvaluator


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--graph-type", type=str, default="fourlang")
    parser.add_argument("-f", "--features", type=str, required=True)
    parser.add_argument("-d", "--dataset-path", type=str, default=None, required=True)
    parser.add_argument("-m", "--mode", type=str, default="predictions")

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s : "
        + "%(module)s (%(lineno)s) - %(levelname)s - %(message)s",
    )

    args = get_args()
    df = pd.read_pickle(args.dataset_path)

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
        assert report, "There are no labels in the dataset, we cannot generate a classification report. Are you evaluating a test set?"
        print(report)



if __name__ == "__main__":
    main()
