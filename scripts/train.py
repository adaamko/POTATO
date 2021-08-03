import argparse
import sys
import logging

from exprel.dataset.hasoc_dataset import HasocDataset
from exprel.dataset.semeval_dataset import SemevalDataset
from exprel.feature_extractor.extract import FeatureExtractor
from exprel.models.model import GraphModel

def load_hasoc(path):
    data = HasocDataset(path)
    extractor = FeatureExtractor(lang="en", cache_fn="en_nlp_cache")
    model = GraphModel()
    graphs = data.parse_graphs(extractor)
    data.set_graphs(graphs)
    data.graphs = graphs
    data.save_graphs("hasoc2020_amr.pickle")
    
def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-g", "--graph-type", type=str, default="fourlang")
    parser.add_argument("-d", "--dataset", type=str, default=None, required=True)
    parser.add_argument("-p", "--dataset-path", type=str, default=None, required=True)
    parser.add_argument("-c", "--graphs-cache", type=str, default=None)
    
    return parser.parse_args()

def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    args = get_args()
    load_hasoc(args.dataset_path)


if __name__ == "__main__":
    main()
