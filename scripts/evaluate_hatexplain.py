from typing import List, Dict
import json
import numpy as np
from pandas import DataFrame
import logging
from argparse import ArgumentParser, ArgumentError
from sklearn.metrics import classification_report
from xpotato.graph_extractor.extract import FeatureEvaluator
from xpotato.dataset.explainable_dataset import ExplainableDataset


def print_classification_report(df: DataFrame, stats: Dict[str, List]):
    print(
        classification_report(
            df.label_id,
            [(n > 0) * 1 for n in np.sum([p for p in stats["Predicted"]], axis=0)],
            digits=3,
        )
    )


def find_good_features(
    feature_file: str,
    train_files: List[str],
    valid_files: List[str],
    save_features: str,
    target: str,
    threshold: float,
) -> None:
    with open(feature_file) as feature_json:
        features = json.load(feature_json)
    if target is None:
        target = list(features.keys())[0]
    if save_features is None:
        logging.warning(
            'No path given for the good features. '
            'They will be saved to this working directory with the name "good_features.json"'
        )
        save_features = "good_features.json"
    if len(train_files) > 1:
        logging.warning(
            "Only the first training file will be used to determine the good features, "
            "but the features will be evaluated on every file given."
        )

    train = ExplainableDataset(path=train_files[0], label_vocab={"None": 0, target: 1})
    train_df = train.to_dataframe()

    evaluator = FeatureEvaluator()
    train_stats = evaluator.evaluate_feature(target, features[target], train_df)[0]
    good_features = []
    bad_features = []
    for (index, stats), feature in zip(train_stats.iterrows(), features[target]):
        if stats["Precision"] >= threshold:
            good_features.append(feature)
        if len(stats["False_positive_indices"]) > len(stats["True_positive_indices"]):
            bad_features.append(feature)
    print(f"Bad features: {len(bad_features)}\n\t{bad_features}")
    print(f"Good features: {len(good_features)}\n\t{good_features}")
    print(f"Train file {train_files[0]} with every feature:")
    print_classification_report(train_df, train_stats)

    with open(save_features, "w") as js:
        json.dump({target: good_features}, js)
    if valid_files is None:
        valid_files = []
    evaluate(feature_file=save_features, files=train_files + valid_files, target=target)


def evaluate(feature_file: str, files: List[str], target: str):
    with open(feature_file) as feature_json:
        features = json.load(feature_json)
    evaluator = FeatureEvaluator()
    if target is None:
        target = list(features.keys())[0]

    for file in files:
        print(f"File: {file}")
        potato = ExplainableDataset(path=file, label_vocab={"None": 0, target: 1})
        df = potato.to_dataframe()
        stats = evaluator.evaluate_feature(target, features[target], df)[0]
        print_classification_report(df, stats)
        print("------------------------")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--mode",
        "-m",
        choices=["find_good_features", "evaluate"],
        help="The mode of operation",
        default="evaluate",
    )
    argparser.add_argument(
        "--features",
        "-f",
        help="Path to the feature to evaluate.Used in both modes",
        required=True,
    )
    argparser.add_argument(
        "--target",
        "-tg",
        help="The target category of your features. If not given, than the code will choose one from the feature file.",
    )
    argparser.add_argument(
        "--threshold",
        "-th",
        help="The minimum precision with which we consider a feature good.",
        default=0.8,
        type=float,
    )
    argparser.add_argument(
        "--train", "-t", help="The train file in potato format", nargs="+"
    )
    argparser.add_argument(
        "--valid", "-v", help="The validation file in potato format", nargs="+"
    )
    argparser.add_argument(
        "--save_features",
        "-sf",
        help="Path to the feature file where the good features will be saved in find_good features mode",
    )
    args = argparser.parse_args()
    if args.mode == "find_good_features":
        if args.train is None:
            raise ArgumentError(
                argument=args.train,
                message="Training file is needed in find_good_features mode",
            )
        find_good_features(
            args.features,
            args.train,
            args.valid,
            args.save_features,
            args.target,
            args.threshold,
        )
    else:
        if args.train is None and args.valid is None:
            raise ArgumentError(
                argument=args.train,
                message="At least one training file or validation is needed in evaluate mode",
            )
        train = [] if args.train is None else args.train
        valid = [] if args.valid is None else args.valid
        evaluate(args.features, train + valid, args.target)
