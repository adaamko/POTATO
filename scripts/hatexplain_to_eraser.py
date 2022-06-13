import json
import os
import csv

import more_itertools as mit

from tuw_nlp.graph.utils import graph_to_pn

save_path='hatexplain/'

if not os.path.exists(save_path+'docs'):
	os.makedirs(save_path+'docs')

###https://github.com/hate-alert/HateXplain/blob/master/Explainability_Calculation_NB.ipynb	
# https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]
            

# used to convert our rationale list to the explanations mask
def rationale_to_explanations(text, rationale_list):
	explanations = []
	#print(rationale_list)
	# make sure it is a list
	if(not isinstance(rationale_list, list)):
		#rationale_list = [rationale_list]
		print(rationale_list, " is not a list")
	for i, word in enumerate(text.split()):
		if word in rationale_list:
			explanations.append(1)
		else:
			explanations.append(0)
	#print(explanations)
	return explanations

def line_to_list(line):
	str = line
	str = str[1:-1]
	str = str.replace('\'', '')
	strlist = str.split(", ")
	return strlist

# Convert dataset into ERASER format: https://github.com/jayded/eraserbenchmark/blob/master/rationale_benchmark/utils.py
def get_evidence(post_id, text, rationale_list):
    output = []
    #explanations = [1,1,0,0,1,1]
    explanations = rationale_to_explanations(text, rationale_list)
    anno_text = text.split()
    indexes = sorted([i for i, each in enumerate(explanations) if each==1])
    span_list = list(find_ranges(indexes))
    #print(([i for i, each in enumerate(explanations) if each==1]))
    for each in span_list:
        if type(each)== int:
            start = each
            end = each+1
        elif len(each) == 2:
            start = each[0]
            end = each[1]+1
        else:
            print('error')

        output.append({"docid":post_id, 
              "end_sentence": -1, 
              "end_token": end, 
              "start_sentence": -1, 
              "start_token": start, 
              "text": ' '.join([str(x) for x in anno_text[start:end]])})
    #if(len(output)>0):
    return output
    #else:
        #return empty_evidence(post_id)
###

def empty_evidence(post_id):
    output = []
    output.append({"docid":post_id, 
        "end_sentence": -1, 
        "end_token": -1, 
        "start_sentence": -1, 
        "start_token": -1, 
        "text": ' '})
    return output

def data_tsv_to_eraser(tsvfile):
	train_tsv = open(tsvfile, encoding="utf8")
	read_tsv = csv.reader(train_tsv, delimiter="\t")
	newname = tsvfile.replace(".tsv", "")
	write_eraser = open(save_path+newname+'.jsonl', 'w')
	# 0 text
	# 1	label
	# 2	label_id
	# 3	rationale
	# 4	graph
	data_set = []
	id = 2
	skip_first = False
	for row in read_tsv:
		if(skip_first == False):
			skip_first = True
			continue
			
		if(row[1] == "None"): # cut out GT None
			id = id+1 # do not forget that
			continue # rip None labels ;(
		
		entry = {}
		#print(row[2])
		id_string = "tsv_line_"+str(id)+".txt"
		write_doc = open(save_path+"docs/"+id_string, 'w', encoding="utf8")
		write_doc.write(row[0]);
		write_doc.close()
		entry['annotation_id'] = id_string
		entry['classification'] = row[1]
		entry['docids'] = 'null'
		entry['evidences'] = [get_evidence(id_string, row[0], line_to_list(row[3]))]
		entry['query'] = "What is the class?"
		entry['query_type'] = None
		data_set.append(entry)
		
		#if(id == 350):
		if(False):
			print(row[0], row[3])
			print(rationale_to_explanations(row[0], line_to_list(row[3])))
			print(get_evidence(id_string, row[0], line_to_list(row[3])))
		
		write_eraser.write(json.dumps(entry)+'\n')
		id = id+1

	train_tsv.close()
	write_eraser.close()

from xpotato.dataset.utils import default_pn_to_graph

def penman_to_nodenames(penman):
	g = default_pn_to_graph(penman)[0]
	return list([(x[1]['name']) for x in g.nodes(data=True)])

#get rationales from subgraph prediction
def get_rationales(datatsvfile, subgraphs):
	train_tsv = open(datatsvfile, encoding="utf8")
	read_tsv = csv.reader(train_tsv, delimiter="\t")
	# 0 text
	# 1	label
	# 2	label_id
	# 3	rationale
	# 4	graph

	# same id as data_tsv to get docs/ filenames right
	id = 2
	skip_first = False
	for row in read_tsv:
		if(skip_first == False):
			skip_first = True
			continue

		subgraphlist = subgraphs[id-2]
		
		if(row[1] == "None"): # cut out GT "None"
			id = id+1 # do not forget that
			continue # rip None labels ;(
			
		predicted_rationales = []
		if(subgraphlist): #check if a rule matched (else predicted rationales are empty list)
			for subgraph in subgraphlist[0]:
				#predicted_rationales.append(penman_to_nodenames(subgraph[0][0])[0]) # get a list
				predicted_rationales.append(penman_to_nodenames(graph_to_pn(subgraph))[0]) # get a list
				
# datatsvfile: the tsv file
# subgraphs: the graphs
# labels: m(xi)j
# labelswithoutr: m(xi\ri)j
# labelsonlyr: m(ri)j
# target: the target class
def prediction_to_eraser(datatsvfile, subgraphs, labels, labelswithoutr, labelsonlyr, target):
	train_tsv = open(datatsvfile, encoding="utf8")
	read_tsv = csv.reader(train_tsv, delimiter="\t")
	newname = datatsvfile.replace(".tsv", "")
	write_eraser = open(save_path+newname+'_prediction.jsonl', 'w')
	# 0 text
	# 1	label
	# 2	label_id
	# 3	rationale
	# 4	graph
	#print(subgraphs)
	data_set = []
	# same id as data_tsv to get docs/ filenames right
	id = 2
	skip_first = False
	for row in read_tsv:
		if(skip_first == False):
			skip_first = True
			continue

		subgraphlist = subgraphs[id-2]
		labellist = labels[id-2]
		labelwithoutr = labelswithoutr[id-2]
		labelonlyr = labelsonlyr[id-2]
		
		if(row[1] == "None"): # cut out GT "None"
			id = id+1 # do not forget that
			continue # rip None labels ;(
		
		label = ""
		if(labellist):
			label = labellist[0] # first label
		if(not label):
			label = "None" # might be a problem with porting
			
		predicted_rationales = []
		if(subgraphlist): #check if a rule matched (else predicted rationales are empty list)
			for subgraph in subgraphlist[0]:
				#predicted_rationales.append(penman_to_nodenames(subgraph[0][0])[0]) # get a list
				predicted_rationales.append(penman_to_nodenames(graph_to_pn(subgraph))[0]) # get a list
		print(predicted_rationales)
		entry = {}
		#print(row[2])
		id_string = "tsv_line_"+str(id)+".txt"
		write_doc = open(save_path+"docs/"+id_string, 'w', encoding="utf8")
		write_doc.write(row[0]);
		write_doc.close()
		entry['annotation_id'] = id_string
		entry['classification'] = label
		# m_xi = m(xi)j
		m_xi = 0
		if(target == label):
			m_xi = 1
		# m_xi_minus_ri = m(xi\ri)j
		m_xi_minus_ri = 0
		if(target == labelwithoutr):
			m_xi_minus_ri = 1
		# m_ri = m(ri)j
		m_ri = 0
		if(target == labelonlyr):
			m_ri = 1
			
		# normal classification: P(target) = m(xi)j
		ptarget = m_xi
		classification_scores_dict = {
			target: ptarget,
			"None": 1-ptarget
		}
		entry['classification_scores'] = classification_scores_dict
		# comprehensiveness = m(xi)j - m(xi\ri)j
		pcomprehensiveness = m_xi - m_xi_minus_ri
		comprehensiveness_classification_scores_dict = {
			target: pcomprehensiveness,
			"None": 1-pcomprehensiveness
		}
		entry['comprehensiveness_classification_scores'] = comprehensiveness_classification_scores_dict
		# sufficiency = m(xi)j - m(ri)j
		psufficiency = m_xi - m_ri
		sufficiency_classification_scores_dict = {
			target: psufficiency,
			"None": 1-psufficiency
		}
		entry['sufficiency_classification_scores'] = sufficiency_classification_scores_dict

		entry['docids'] = 'null'
		rationales_dict = {
		  "docid": id_string,
		  "hard_rationale_predictions": get_evidence(id_string, row[0], predicted_rationales)
		}
		entry['rationales'] = [rationales_dict]
		entry['query'] = "What is the class?"
		entry['query_type'] = None
		data_set.append(entry)
		
		#if(id == 350):
		if(False):
			print(row[0], row[3])
			print(rationale_to_explanations(row[0], line_to_list(row[3])))
			print(get_evidence(id_string, row[0], line_to_list(row[3])))
		
		write_eraser.write(json.dumps(entry)+'\n')
		id = id+1

	train_tsv.close()
	write_eraser.close()