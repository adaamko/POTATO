import argparse
import logging
import pickle
import sys
import json
from collections import defaultdict

import networkx as nx
import pandas as pd
from exprel.dataset.hasoc_dataset import HasocDataset
from exprel.dataset.utils import amr_pn_to_graph
from networkx.algorithms.isomorphism import DiGraphMatcher
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from tuw_nlp.graph.utils import GraphFormulaMatcher, pn_to_graph


def evaluate_features(dataset, graphs_path, features, graph_format):
    labels = dataset.label.tolist()
    with open(graphs_path, "rb") as f:
        graphs = pickle.load(f)

    matches = []
    predicted = []
    labels = []

    feature_values = []
    for k in features:
        for f in features[k]:
            feature_values.append(f)
    if graph_format == "amr":
        matcher = GraphFormulaMatcher(feature_values, converter=amr_pn_to_graph)
    else:
        matcher = GraphFormulaMatcher(feature_values, converter=pn_to_graph)

    for i, g in tqdm(enumerate(graphs)):
        feats = matcher.match(g)
        for key, feature in feats:
            matches.append(features[key][feature][0])
            predicted.append(key)
            break
        else:
            matches.append("")
            predicted.append("NOT")

    d = {"Sentence": dataset.sentence.tolist(), "Predicted label": predicted, "Matched rule": matches}
    df = pd.DataFrame(d)
    return df


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--graph-type", type=str, default="fourlang")
    parser.add_argument("-f", "--features", type=str, required=True)
    parser.add_argument("-d", "--dataset-path", type=str,
                        default=None, required=True)
    parser.add_argument("-g", "--input-graphs", type=str,
                        default=None, required=True)

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    args = get_args()
    df = pd.read_csv(args.dataset_path, delimiter="\t")
    data = HasocDataset(df)
    df = data.to_dataframe()
    df = df.rename(columns={'preprocessed_text': 'sentence', 'task1': 'label'})
    with open(args.features) as f:
        features = json.load(f)
    df = evaluate_features(df, args.input_graphs, features, args.graph_type)
    df.to_csv(sys.stdout, sep='\t')


if __name__ == "__main__":
    main()
