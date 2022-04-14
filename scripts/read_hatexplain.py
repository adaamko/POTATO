import json
import numpy as np


def read_json(file_path):
    data_by_target = {}
    no_rational_by_target = {}
    with open(file_path) as dataset:
        data = json.load(dataset)
        for post in data.values():
            sentence = " ".join(post["post_tokens"])
            targets = {}
            labels = {}
            for annotation in post["annotators"]:
                if annotation["label"] not in labels:
                    labels[annotation["label"]] = 1
                else:
                    labels[annotation["label"]] += 1
                for target in annotation["target"]:
                    if target not in targets:
                        targets[target] = 1
                    else:
                        targets[target] += 1
            if len(labels) != len(post["annotators"]):
                label = max(labels.items(), key=lambda x: x[1])[0]
                target = [t[0] for t in targets.items() if t[1] > 1]
            rationale = []
            if len(post["rationales"]) > 0:
                rats = [n for n in post["rationales"] if len(n) == len(post["post_tokens"])]
                rationale = np.round(np.mean(rats, axis=0), decimals=0).tolist()
            if len(target) == 1:
                if len(post["rationales"]) == 0 and target[0] != "None":
                    if target[0] not in no_rational_by_target:
                        no_rational_by_target[target[0]] = 0
                    no_rational_by_target[target[0]] += 1
                if target[0] not in data_by_target:
                    data_by_target[target[0]] = []
                data_by_target[target[0]].append({"tokens": post["post_tokens"],
                                                  "sentence": sentence,
                                                  "rationale": rationale})
    return data_by_target


if __name__ == '__main__':
    dt_by_target = read_json("dataset.json")
    for name, list_of_dicts in dt_by_target.items():
        with open(f"{name.lower()}.json", "w") as json_file:
            json.dump(list_of_dicts, json_file, indent=4)
