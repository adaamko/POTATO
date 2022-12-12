import os

from xpotato.dataset.utils import default_pn_to_graph
from xpotato.graph_extractor.extract import FeatureEvaluator
from xpotato.graph_extractor.rule import Rule, RuleSet

dir_name = os.path.dirname(os.path.realpath(__file__))

FEATURE1 = [["(u_0 / find :obl (u_1 / .*) :obj (u_2 / .*))"], [], "FIND"]
FEATURE2 = [["(u_0 / write :nsubj (u_1 / .*) :obj (u_2 / .*))"], [], "WRITE"]

FEATURE3 = [
    ["(u_0 / find :obl (u_1 / .*) :obj (u_2 / .*))"],
    [],
    "FIND",
    [{"ARG1": 1, "ARG2": 2}],
]
FEATURE4 = [
    ["(u_0 / write :nsubj (u_1 / .*) :obj (u_2 / .*))"],
    [],
    "WRITE",
    [{"ARG1": 1, "ARG2": 2}],
]

FEATURE_DICT = {
    "FIND": [[["(u_0 / find :obl (u_1 / .*) :obj (u_2 / .*))"], [], "FIND"]],
    "WRITE": [[["(u_0 / write :nsubj (u_1 / .*) :obj (u_2 / .*))"], [], "WRITE"]],
}

GRAPH = "(u_2 / write  :nsubj (u_1 / person)  :obj (u_4 / sentence  :det (u_3 / this))  :conj (u_6 / find  :cc (u_5 / and)  :obj (u_7 / object)  :obl (u_9 / location  :case (u_8 / in)))  :punct (u_10 / PERIOD)  :root-of (u_0 / root))"


def test_rule():
    rule1 = Rule(FEATURE1)

    assert rule1.to_list() == FEATURE1


def test_ruleset_to_list():
    rule_set = RuleSet([Rule(FEATURE1)])

    rule_set.add_rule(Rule(FEATURE2))

    assert rule_set.to_list() == [FEATURE1, FEATURE2]


def test_ruleset_to_dict():
    rule_set = RuleSet([Rule(FEATURE1), Rule(FEATURE2)])

    assert rule_set.to_dict() == {"FIND": [FEATURE1], "WRITE": [FEATURE2]}


def test_ruleset_from_dict_to_list():
    rule_set = RuleSet()
    rule_set.from_dict(FEATURE_DICT)

    assert rule_set.to_list() == [FEATURE1, FEATURE2]


def test_ruleset_json():
    rule_set = RuleSet()

    rule_set.from_json(os.path.join(dir_name, "features.json"))

    assert rule_set.to_list() == [FEATURE1, FEATURE2]


def test_ruleset_to_tsv():
    rule_set = RuleSet([Rule(FEATURE1), Rule(FEATURE2)])

    rule_set.to_tsv(os.path.join(dir_name, "features.tsv"))

    rule_set = RuleSet()

    rule_set.from_tsv(os.path.join(dir_name, "features.tsv"))

    assert rule_set.to_list() == [FEATURE1, FEATURE2]


def test_ruleset_openie():
    rule_set = RuleSet([Rule(FEATURE3, openie=True), Rule(FEATURE4, openie=True)])
    rule_set.to_tsv(os.path.join(dir_name, "features_openie.tsv"))

    rule_set = RuleSet()
    rule_set.from_tsv(os.path.join(dir_name, "features_openie.tsv"))

    assert rule_set.to_list() == [FEATURE3, FEATURE4]


def test_openie_matching():
    evaluator = FeatureEvaluator()

    G, _ = default_pn_to_graph(GRAPH)

    rule_set = RuleSet([Rule(FEATURE3, openie=True), Rule(FEATURE4, openie=True)])

    triplets = list(evaluator.annotate(G, rule_set.to_list()))

    assert triplets == [
        {"relation": "FIND", "ARG1": "location", "ARG2": "object"},
        {"relation": "WRITE", "ARG1": "person", "ARG2": "sentence"},
    ]
