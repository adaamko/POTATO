from tabulate import tabulate 
import json

datadir="./hatexplain"

with open(datadir+'/eraser_output.json') as fp:
	output_data = json.load(fp)
	
output = []

output.append(['Plausibility'])
if 'iou_scores' in output_data:
	output.append(['IOU F1 :', round(output_data['iou_scores'][0]['macro']['f1'], 3)])
	output.append(['Token F1 :', round(output_data['token_prf']['instance_macro']['f1'], 3)])

if 'token_soft_metrics' in output_data:
	output.append(['AUPRC :', round(output_data['token_soft_metrics']['auprc'], 3)])

output.append(['Faithfulness'])
if 'classification_scores' in output_data:
	output.append(['Comprehensiveness :', round(output_data['classification_scores']['comprehensiveness'], 3)])
	output.append(['Sufficiency :', round(output_data['classification_scores']['sufficiency'], 3)])
else:
	output.append('--')
#output.append("")

#print(tabulate(output))
print("")
print("\\begin{table}")
print("\label{tab:example}")
print("\centering")
print(tabulate(output, headers=["Metric", "Value"], tablefmt="latex"))
print("\caption{Latex ERASER Table Example}")
print("\end{table}")