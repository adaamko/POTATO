
import argparse
import logging
import json
import sys
import stanza
import networkx as nx

from tuw_nlp.grammar.text_to_4lang import TextTo4lang
from tuw_nlp.text.pipeline import CachedStanzaPipeline
from tuw_nlp.graph.utils import graph_to_isi, pn_to_graph, graph_to_pn
from tuw_nlp.graph.utils import GraphMatcher
from tqdm import tqdm


class GraphExtractor():

    def __init__(
            self, cache_dir=None, cache_fn=None, lang=None):
        self.cache_dir = cache_dir
        self.cache_fn = cache_fn
        self.lang = lang
        self.nlp = None
        self.matcher = None

    def init_nlp(self):
        self.nlp = stanza.Pipeline('en')

    def set_matcher(self, patterns):
        self.matcher = GraphMatcher(patterns)

    def parse_iterable(self, iterable):
        with TextTo4lang(lang=self.lang, nlp_cache=self.cache_fn, cache_dir=self.cache_dir) as tfl:
            for sen in tqdm(iterable):
                fl_graphs = list(tfl(sen))
                g = fl_graphs[0]
                for n in fl_graphs[1:]:
                    g = nx.compose(g, n)
                yield g
