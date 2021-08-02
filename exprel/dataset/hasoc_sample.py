import networkx as nx
from bs4 import BeautifulSoup
from exprel.dataset.sample import Sample
from tuw_nlp.text.utils import preprocess_tweet
from exprel.dataset.utils import amr_pn_to_graph


class HasocSample(Sample):
    def __init__(self, tweet_id, text, task1, task2, ID, nlp, graph_format="amr"):
        super().__init__()
        self.hasoc_id = ID
        self.tweet_id = tweet_id
        self.original_text = text
        self.task1 = task1
        self.task2 = task2
        self.preprocessed_text = self.prepare_sentence(text)
        self.nlp = nlp
        self.graph = None
        self.doc = None
        self.graph_format = graph_format
        self.prepare_doc()

    def prepare_sentence(self, tweet):
        return preprocess_tweet(tweet, keep_hashtag=False, keep_username=False)

    def _postprocess(self, graph):    
        return graph

    def set_graph(self, graph):
        self.graph = graph

    def prepare_doc(self):
        self.doc = self.nlp(self.preprocessed_text)
        if self.graph_format == "amr":
            graphs = self.doc._.to_amr()
            G = None
            for graph in graphs:
                G, _ = amr_pn_to_graph(graph)
            if G:
                self.graph = G