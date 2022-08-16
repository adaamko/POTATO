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

def check_if(limit, num, frame, frame_without_rationales):
	row = frame.iloc[num]
	row2 = frame_without_rationales.iloc[num]
	
	if(limit == "NO"):
		return True
	
	# GT, predicted
	if(limit == "TP" and str(row2["label"])!="None" and str(row["Predicted label"])!="nan"):
		return True
		
	if(limit == "FP" and str(row2["label"])=="None" and str(row["Predicted label"])!="nan"):
		return True
		
	if(limit == "TN" and str(row2["label"])=="None" and str(row["Predicted label"])=="nan"):
		return True

	if(limit == "FN" and str(row2["label"])!="None" and str(row["Predicted label"])=="nan"):
		return True
		
	return False
	
def get_confusion_class(num, frame, frame_without_rationales):
	row = frame.iloc[num]
	row2 = frame_without_rationales.iloc[num]
	
	# GT, predicted
	if(str(row2["label"])!="None" and str(row["Predicted label"])!="nan"):
		return "TP"
		
	if(str(row2["label"])=="None" and str(row["Predicted label"])!="nan"):
		return "FP"
		
	if(str(row2["label"])=="None" and str(row["Predicted label"])=="nan"):
		return "TN"

	if(str(row2["label"])!="None" and str(row["Predicted label"])=="nan"):
		return "FN"


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == "__main__":
	argparser = ArgumentParser()

	numarg = argparser.add_argument(
		"--limit",
		"-l",
		help="Limit sample type to TP, FP, TN, FN.",
		choices=["NO", "TP", "FP", "TN", "FN"],
		default="NO",
		type=str,
	)

	numarg = argparser.add_argument(
		"--number",
		"-n",
		help="The number of examples to inspect.",
		default=10,
		type=int,
	)
	
	numarg = argparser.add_argument(
		"--page",
		"-p",
		help="Not the first but p-th n samples.",
		default=1,
		type=int,
	)
	
	numarg = argparser.add_argument(
		"--random",
		"-r",
		help="Not first n samples but n random samples.",
		action='store_true',
	)
	
	numarg = argparser.add_argument(
		"--seed",
		"-s",
		help="Random seed for random mode.",
		default=0,
		type=int,
	)
	#argparser.add_argument(
	#    "--train", "-t", help="The train file in potato format", nargs="+"
	#)

	args = argparser.parse_args()
	
	if(args.random):
		random.seed(args.seed)
	
	outtext = " "
	if(args.limit != "NO"):
		outtext = outtext + args.limit + " "
	if(args.random):
		outtext = outtext + "random "
	print("Printing "+str(args.number)+ outtext+"examples...")
	print("------------------------")
	numbers = []
	
	frame = None
	try: 
		frame = pandas.read_csv("temp_matched_result.tsv", sep="\t")
	except IOError as e:
		print("Evaluation dataframe files not found! Please run evaluate_hatexplain.py first.")
	frame_without_rationales = pandas.read_csv("temp_df_without_rationales.tsv", sep="\t")
	if(args.number > frame.shape[0]):
		raise ArgumentError(
			message="Number is bigger than dataframe rows!",
			argument=numarg
			)
			
	num = -1
	random_attempts=0
	for i in range(args.number*args.page):
		fitting_num = False
		
		# choose fitting num
		while(fitting_num == False):
			# choose random num
			if(args.random):
				num = random.randint(0, frame.shape[0]-1)
				while(num in numbers):
					num = random.randint(0, frame.shape[0]-1)
					random_attempts=random_attempts+1
				if(random_attempts >= 10*frame.shape[0]):
					print("No more examples found after "+str(random_attempts-1)+" random attempts.")
					print("------------------------")
					break
			# choose next num
			else:
				num = num+1
				if(num >= frame.shape[0]):
					print("No more examples found.")
					print("------------------------")
					break
						
			if(check_if(args.limit, num, frame, frame_without_rationales)):
				fitting_num = True
			
		# no more fitting nums
		if(fitting_num == False):
			break

		# page
		if(i < args.number*(args.page-1)):
			continue

		numbers.append(num)
		
		# select num
		row = frame.iloc[num]
		row2 = frame_without_rationales.iloc[num]
		print(bcolors.HEADER+"Sentence: ("+str(num)+")"+bcolors.ENDC)
		print(row["Sentence"])
		print("Labels:")
		print("GT: " + bcolors.OKBLUE+str(row2["label"])+bcolors.ENDC + " vs Prediction: "+bcolors.OKBLUE+str(row["Predicted label"])+bcolors.ENDC+" (it's "+get_confusion_class(num, frame, frame_without_rationales)+")")
		print("Matched rule: "+ str(row["Matched rule"]))
		print("Rationals:")
		print("GT: " + bcolors.OKBLUE+row2["rationale_lemma"]+bcolors.ENDC + " vs Prediction: "+bcolors.OKBLUE+row["Predicted rational"]+bcolors.ENDC)
		print("------------------------")
		#print(num)
		

