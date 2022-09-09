from typing import List, Dict
import json
import numpy as np
from pandas import DataFrame
import pandas
import logging
from argparse import ArgumentParser, ArgumentError
from sklearn.metrics import classification_report, confusion_matrix
from xpotato.graph_extractor.extract import FeatureEvaluator
from xpotato.dataset.explainable_dataset import ExplainableDataset
from xpotato.dataset.utils import save_dataframe
from tuw_nlp.common.eval import *

from hatexplain_to_eraser import data_tsv_to_eraser, prediction_to_eraser, get_rationales
from call_eraser import call_eraser

def print_classification_report(df: DataFrame, stats: Dict[str, List]):
    #print([(n > 0) * 1 for n in np.sum([p for p in stats["Predicted"]], axis=0)])
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
            "No path given for the good features. "
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

def remove_rationals(x, rationals):
    text = x
    for rational in rationals:
        text = text.replace(rational, "")
    return text

def concat_rationals(rationals):
    if(not rationals):
        return "UNK"
    text = ""
    for rational in rationals:
        text = text + rational + " "
    return text

def evaluate(feature_file: str, files: List[str], target: str):
    #feature_file="sexism_rules.json"
    with open(feature_file) as feature_json:
        features = json.load(feature_json)
    evaluator = FeatureEvaluator()
    if target is None:
        target = list(features.keys())[0]
    
    for file in files:
        #file="women/minority_val_pure.tsv"
        #target="Women"
        potato = ExplainableDataset(path=file, label_vocab={"None": 0, target: 1})
        df = potato.to_dataframe()
        stats = evaluator.evaluate_feature(target, features[target], df)[0]

        # get labels and predicted rationals from matched_results
        matched_result = evaluator.match_features(df, features[target], multi=True, return_subgraphs=True, allow_multi_graph=True)
        subgraphs = matched_result["Matched subgraph"]
        labels = matched_result["Predicted label"]
        rationale_as_text_list = get_rationales(file, subgraphs)
        matched_result["Predicted rational"] = rationale_as_text_list
        matched_result.to_csv('temp_matched_result.tsv', sep="\t")
        
        # get labels by removing the predicted rationals
        df_without_rationales = df.copy()
        for i in range(df_without_rationales['text'].size):
            df_without_rationales['text'][i] = remove_rationals(df_without_rationales['text'][i], rationale_as_text_list[i])
        save_dataframe(df_without_rationales, 'temp_df_without_rationales.tsv')
        potato_without_rationales = ExplainableDataset(path='temp_df_without_rationales.tsv', label_vocab={"None": 0, target: 1})
        potato_without_rationales.set_graphs(potato_without_rationales.parse_graphs(graph_format="ud"))
        df_without_rationales = potato_without_rationales.to_dataframe()
        save_dataframe(df_without_rationales, 'temp_df_without_rationales.tsv') # to view graphs
        matched_result = evaluator.match_features(df_without_rationales, features[target], multi=True, return_subgraphs=True, allow_multi_graph=True)
        labels_without_rationales = matched_result["Predicted label"]

        # get labels with only the rationals
        df_only_rationales = df.copy()
        for i in range(df_only_rationales['text'].size):
            df_only_rationales['text'][i] = concat_rationals(rationale_as_text_list[i])
        save_dataframe(df_only_rationales, 'temp_df_only_rationales.tsv')
        potato_only_rationales = ExplainableDataset(path='temp_df_only_rationales.tsv', label_vocab={"None": 0, target: 1})
        potato_only_rationales.set_graphs(potato_only_rationales.parse_graphs(graph_format="ud"))
        df_only_rationales = potato_only_rationales.to_dataframe()
        save_dataframe(df_only_rationales, 'temp_df_only_rationales.tsv') # to view graphs
        matched_result = evaluator.match_features(df_only_rationales, features[target], multi=True, return_subgraphs=True, allow_multi_graph=True)
        labels_only_rationales = matched_result["Predicted label"]

        # convert the data to eraser format and run eraser
        data_tsv_to_eraser(file)
        prediction_to_eraser(file, rationale_as_text_list, labels, labels_without_rationales, labels_only_rationales, target)
        call_eraser("None", "./hatexplain", "val", "./hatexplain/val_prediction.jsonl")
        print("------------------------")
        #print_classification_report(df, stats)
        #print("------------------------")

        tn0, fp0, fn0, tp0 = confusion_matrix(1-df.label_id, [1-(n > 0) * 1 for n in np.sum([p for p in stats["Predicted"]], axis=0)]).ravel()
        confusion_dict_neutral={"TN":tn0,"FP":fp0,"FN":fn0,"TP":tp0}
        tn1, fp1, fn1, tp1 = confusion_matrix(df.label_id, [(n > 0) * 1 for n in np.sum([p for p in stats["Predicted"]], axis=0)]).ravel()
        confusion_dict_target={"TN":tn1,"FP":fp1,"FN":fn1,"TP":tp1}
        cat_stats={target : confusion_dict_target, "None" : confusion_dict_neutral}
        print_cat_stats(cat_stats) #s,tablefmt="latex_booktabs"

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
