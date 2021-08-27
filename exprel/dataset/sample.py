
class Sample:
    def __init__(self, example):
        self.text = example[0]
        self.label = example[1]
        self.graph = None

    def set_graph(self, graph):
        self.graph = graph