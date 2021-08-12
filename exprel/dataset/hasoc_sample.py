import networkx as nx
from bs4 import BeautifulSoup
from exprel.dataset.sample import Sample
from tuw_nlp.text.utils import preprocess_tweet
from exprel.dataset.utils import amr_pn_to_graph


class HasocSample(Sample):
    def __init__(self, text, task1, task2, ID, nlp, graph_format="amr"):
        super().__init__()
        self.hasoc_id = ID
        self.original_text = text
        self.task1 = task1
        self.task2 = task2
        self.preprocessed_text = text
        self.nlp = nlp
        self.graph = None
        self.doc = None
        self.graph_format = graph_format

    def prepare_sentence(self, tweet):
        return preprocess_tweet(tweet, keep_hashtag=False, keep_username=False)

    def _postprocess(self, graph):
        return graph

    def set_graph(self, graph):
        self.graph = graph
