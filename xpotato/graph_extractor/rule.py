class Rule:
    def __init__(self, rule):
        self.positive_samples = rule[0]
        self.negative_samples = rule[1]
        self.label = rule[2]
