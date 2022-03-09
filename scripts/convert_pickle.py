import argparse
import json
import logging
import sys

from xpotato.dataset.dataset import Dataset
from xpotato.dataset.utils import save_dataframe


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-p", "--pickle", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()

    path = args.pickle
    output = args.output

    dataset = Dataset(path=path, binary=True)
    df = dataset.to_dataframe()

    save_dataframe(df, output)


if __name__ == "__main__":
    main()
