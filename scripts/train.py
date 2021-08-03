import argparse
import sys
import logging

from exprel.dataset.hasoc_dataset import HasocDataset
from exprel.dataset.semeval_dataset import SemevalDataset


def get_args():
    parser = argparse.ArgumentParse(description="")
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



if __name__ == "__main__":
    main()