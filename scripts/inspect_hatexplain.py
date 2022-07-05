from typing import List, Dict
import json
import random
import numpy as np
from pandas import DataFrame
import pandas
import logging
from argparse import ArgumentParser, ArgumentError
from sklearn.metrics import classification_report
from xpotato.graph_extractor.extract import FeatureEvaluator
from xpotato.dataset.explainable_dataset import ExplainableDataset
from xpotato.dataset.utils import save_dataframe

from hatexplain_to_eraser import data_tsv_to_eraser, prediction_to_eraser, get_rationales
from call_eraser import call_eraser



if __name__ == "__main__":
	argparser = ArgumentParser()

	numarg = argparser.add_argument(
		"--number",
		"-n",
		help="The number of random examples to inspect.",
		default=10,
		type=int,
	)
	#argparser.add_argument(
	#    "--train", "-t", help="The train file in potato format", nargs="+"
	#)

	args = argparser.parse_args()
	print("Printing "+str(args.number)+" random examples...")
	print("------------------------")
	numbers = []
	frame = pandas.read_csv("temp_matched_result.tsv", sep="\t")
	frame_without_rationales = pandas.read_csv("temp_df_without_rationales.tsv", sep="\t")
	if(args.number > frame.shape[0]):
		raise ArgumentError(
			message="number is bigger than dataframe rows",
			argument=numarg
			)
			
			
	for i in range(args.number):
		num = random.randint(0, frame.shape[0])
		while(num in numbers):
			num = random.randint(0, frame.shape[0])
		numbers.append(num)
		#print(frame[num])
		row = frame.iloc[num]
		row2 = frame_without_rationales.iloc[num]
		print("Sentence:")
		print(row["Sentence"])
		print("Labels:")
		print("GT: " + str(row2["label"]) + " vs Prediction: "+str(row["Predicted label"]))
		print("Matched rule: "+ str(row["Matched rule"]))
		print("Rationals:")
		print("GT: " + row2["rationale"] + " vs Prediction: "+row["Predicted rational"])
		print("------------------------")
		#print(num)