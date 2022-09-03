# potato
function evaluate() {
	path=$1
	rules=$2

	# process path to folder
	foldersign="/"
	minus="-"
	foldername=${path//$foldersign/$minus}
	tsv=".tsv"
	empty=""
	foldername=${foldername//$tsv/$empty}

	mkdir $foldername
	cp $path val.tsv
	python evaluate_hatexplain.py -f $rules -t val.tsv | tee out.txt
	cp temp_matched_result.tsv $foldername
	cp temp_df_without_rationales.tsv $foldername
	cp temp_df_only_rationales.tsv $foldername
	mv out.txt $foldername
	rm val.tsv
}

#evaluate women/minority_val_all.tsv sexism_rules.json
#evaluate women/majority_val_all.tsv sexism_rules.json
#evaluate homosexual/minority_val_all.tsv homophobia_rules.json
#evaluate homosexual/majority_val_all.tsv homophobia_rules.json

evaluate women/minority_val_one_majority.tsv sexism_rules.json
evaluate women/majority_val_one_majority.tsv sexism_rules.json
evaluate homosexual/minority_val_one_majority.tsv homophobia_rules.json
evaluate homosexual/majority_val_one_majority.tsv homophobia_rules.json

evaluate women/minority_val_pure.tsv sexism_rules.json
evaluate women/majority_val_pure.tsv sexism_rules.json
evaluate homosexual/minority_val_pure.tsv homophobia_rules.json
evaluate homosexual/majority_val_pure.tsv homophobia_rules.json

