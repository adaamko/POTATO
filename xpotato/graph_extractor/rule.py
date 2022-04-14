import json
from typing import Dict, List, Union


class Rule:
    def __init__(self, rule, openie=False):
        self.positive_samples = rule[0]
        self.negative_samples = rule[1]
        self.label = rule[2]
        self.openie = openie

        self.marked_nodes = None
        if openie:
            self.marked_nodes = rule[3]

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Rule):
            return False
        return (
            sorted(self.positive_samples) == sorted(__o.positive_samples)
            and sorted(self.negative_samples) == sorted(__o.negative_samples.sort())
            and self.label == __o.label
        )

    def to_list(self) -> List[List[Union[List[str], str]]]:
        return (
            [self.positive_samples, self.negative_samples, self.label]
            if self.openie == False
            else [
                self.positive_samples,
                self.negative_samples,
                self.label,
                self.marked_nodes,
            ]
        )


class RuleSet:
    def __init__(self, rules: List[Rule] = None):
        if rules is None:
            self.rules = []
        else:
            self.rules = rules

    def __iter__(self):
        for rule in self.rules:
            yield rule

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, RuleSet):
            return False
        return self.rules == __o.rules

    def to_tsv(self, tsv_path: str):
        with open(tsv_path, "w") as f:
            for rule in self.rules:
                positive_samples = ";".join(rule.positive_samples)
                negative_samples = ";".join(rule.negative_samples)
                label = rule.label
                marked_nodes = rule.marked_nodes

                rule_str = f"{label}\t{positive_samples}\t{negative_samples}"

                if rule.openie:
                    rule_str += f"\t{json.dumps(marked_nodes)}"
                f.write(rule_str + "\n")

    def from_tsv(self, tsv_path: str):
        with open(tsv_path, "r") as f:
            for line in f:
                line = line.strip("\n")
                line = line.split("\t")

                positive_samples = [] if line[1] == "" else line[1].split(";")
                negative_samples = [] if line[2] == "" else line[2].split(";")
                label = line[0].strip()
                rule = None
                if len(line) == 3:
                    rule = Rule(
                        [positive_samples, negative_samples, label], openie=False
                    )
                elif len(line) == 4:
                    marked_nodes = [] if line[3] == "" else json.loads(line[3])
                    rule = Rule(
                        [positive_samples, negative_samples, label, marked_nodes],
                        openie=True,
                    )
                else:
                    raise Exception(f"Invalid number of fields: {line}")

                if not rule:
                    raise Exception(f"Invalid rule: {line}")
                self.add_rule(rule)

    def from_dict(
        self, rules: Dict[str, List[List[Union[List[str], str]]]], openie: bool = False
    ):
        for key, value in rules.items():
            for rule in value:
                self.add_rule(Rule(rule, openie=openie))

    def from_json(self, json_path: str, openie: bool = False):
        with open(json_path, "r") as f:
            rules = json.load(f)

        for key, value in rules.items():
            for rule in value:
                self.add_rule(Rule(rule, openie=openie))

    def to_json(self, json_path: str):
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f)

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def to_dict(self) -> Dict[str, List[List[Union[List[str], str]]]]:
        rule_dict = {rule.label: [] for rule in self.rules}

        for rule in self.rules:
            rule_dict[rule.label].append(rule.to_list())

        return rule_dict

    def to_list(self) -> List[List[Union[List[str], str]]]:
        return [rule.to_list() for rule in self.rules]
