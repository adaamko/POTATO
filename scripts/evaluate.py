import argparse
import logging
import pickle
import sys
import json
from collections import defaultdict

import networkx as nx
import pandas as pd
from xpotato.dataset.utils import default_pn_to_graph
from xpotato.graph_extractor.extract import FeatureEvaluator


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--graph-type", type=str, default="fourlang")
    parser.add_argument("-f", "--features", type=str, required=True)
    parser.add_argument("-d", "--dataset-path", type=str, default=None, required=True)

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
    if "label" in df:
        pred_df["label"] = df.label
    pred_df.to_csv(sys.stdout, sep="\t")


if __name__ == "__main__":
    main()
