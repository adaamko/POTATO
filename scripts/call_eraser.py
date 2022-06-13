import os, sys
file_path = 'eraserbenchmark/'
sys.path.append(os.path.dirname(file_path))

import eraserbenchmark.rationale_benchmark.metrics as eb

import contextlib, logging

class DiscardEraserBenchMarkStdOut(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DiscardEraserBenchMarkStdOut()
    yield
    sys.stdout = save_stdout

#--data_dir : Location of the folder which contains the dataset in eraser format
#--results : The location of the model output file in eraser format
#--score_file : The file name and location to write the output

def call_eraser(datadir, testtrainorval, pathtopredictions, silent=False):
	import sys, os
	pkgpath = os.getcwd()+"\\eraserbenchmark"
	print(pkgpath)
	sys.path.append(pkgpath)
	import rationale_benchmark.metrics as eraser
	if silent:
		logger = logging.getLogger()
		logger.disabled = True
	#dir(rationale_benchmark.metrics)
	eraser.runEvaluation("None", # neutralclassname
						data_dir=datadir, # data dir
						split=testtrainorval, # split
						results=pathtopredictions, # results 
						score_file=datadir+"/eraser_output.json", # score
						strict=False) # strict
						#iou_thresholds=[0.5], # iou
						#aopc_thresholds=[0.01, 0.05, 0.1, 0.2, 0.5]) # aopc
	if silent:
		logger.disabled = False
	print_eraser_results(datadir)

"""
def call_eraser(datadir, testtrainorval, pathtopredictions):
	#args = ['--split', 'test', '--strict', '--data_dir', 'movies', '--results', './movies/movies_majority_human_perf.jsonl']
	args = ['--split', testtrainorval, '--data_dir', datadir, '--results', pathtopredictions, '--score_file', 'eraser_output.json']
	for arg in args:
		sys.argv.append(arg)
		
	# suppress text
	SUPPRESS_TEXT = False

	if SUPPRESS_TEXT:
		logger = logging.getLogger()
		logger.disabled = True
		with nostdout():
			eb.main()
		logger.disabled = False
	else:
		eb.main()
	print_eraser_results()
"""

def print_eraser_results(datadir):
	# print the required results
	import json
	with open(datadir+'/eraser_output.json') as fp:
		output_data = json.load(fp)

	print('\nPlausibility')
	if 'iou_scores' in output_data:
		print('IOU F1 :', output_data['iou_scores'][0]['macro']['f1'])
		print('Token F1 :', output_data['token_prf']['instance_macro']['f1'])
	
	if 'token_soft_metrics' in output_data:
		print('AUPRC :', output_data['token_soft_metrics']['auprc'])

	print('\nFaithfulness')
	if 'classification_scores' in output_data:
		print('Comprehensiveness :', output_data['classification_scores']['comprehensiveness'])
		print('Sufficiency :', output_data['classification_scores']['sufficiency'])
	else:
		print('--')
		
if __name__ == "__main__":
	call_eraser("./hatexplain", "val", "./hatexplain/val_prediction.jsonl")