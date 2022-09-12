import json
import argparse
from tuw_nlp.common.eval import *

my_parser = argparse.ArgumentParser(description='Which model to use')

# Add the arguments
my_parser.add_argument('foldername',
                        metavar='--foldername',
                        nargs='?',
                        type=str,
                        default="evaluation",
                        help='foldername of the data for description')

args = my_parser.parse_args()

# print the required results
with open('../eraser_output.json') as fp:
    output_data = json.load(fp)

print("\n"+args.foldername+":")
print("------------------------")
	
print('Plausibility')
if 'iou_scores' in output_data:
	print('IOU F1 :', round(output_data['iou_scores'][0]['macro']['f1'], 3))
	print('( P :', round(output_data['iou_scores'][0]['macro']['p'], 3), ', R :', round(output_data['iou_scores'][0]['macro']['r'], 3), ')')
	print('Token F1 :', round(output_data['token_prf']['instance_macro']['f1'], 3))
	print('( P :', round(output_data['token_prf']['instance_macro']['p'], 3), ', R :', round(output_data['token_prf']['instance_macro']['r'], 3), ')')
	
if 'token_soft_metrics' in output_data:
	print('AUPRC :', round(output_data['token_soft_metrics']['auprc'], 3))

print('\nFaithfulness')
if 'classification_scores' in output_data:
	print('Comprehensiveness :', round(output_data['classification_scores']['comprehensiveness'], 3))
	print('Sufficiency :', round(output_data['classification_scores']['sufficiency'], 3))
else:
	print('--')
print("")

print("------------------------")

with open('../cat_stats.json') as fp:
    cat_stats = json.load(fp)
    print_cat_stats(cat_stats) #s,tablefmt="latex_booktabs"