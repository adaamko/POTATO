
import argparse
import logging
import json
import sys
import stanza
import networkx as nx

from tuw_nlp.grammar.ud_fl import UD_Fourlang
from tuw_nlp.graph.utils import graph_to_isi, pn_to_graph, graph_to_pn
from tuw_nlp.graph.utils import GraphMatcher


class FeatureExtractor():
    def __init__(
            self, cache_dir=None, ud_fl_cache_fn=None, fl_attr_cache_fn=None):
        self.ud_fl = UD_Fourlang(cache_dir=cache_dir, cache_fn=ud_fl_cache_fn)
        self.nlp = None
        self.matcher = None

    def init_nlp(self):
        self.nlp = stanza.Pipeline('en')

    def set_matcher(self, patterns):
        self.matcher = GraphMatcher(patterns)

    def parse(self, text):
        if self.nlp is None:
            self.init_nlp()

        return self.nlp(text)

    def get_fl(self, sen):
        fl = self.ud_fl.parse(sen, 'ud', 'fourlang', 'amr-sgraph-src')
        return fl

    def parse_text(self, sentence):
        doc = self.parse(sentence)
        fl = self.get_fl(doc.sentences[0])

        return fl

    def extract(self, sen):
        doc = self.parse(sen)
        fl = self.get_fl(doc.sentences[0])

        if fl:
            output, root = pn_to_graph(fl)

            return output, root

        else:
            return nx.DiGraph(), None


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-cd", "--cache-dir", default=None, type=str)
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")
    args = get_args()
    extractor = FeatureExtractor(cache_dir=args.cache_dir)
    for sen in sys.stdin:
        doc = extractor.parse(sen)
        fl = extractor.get_fl(doc.sentences[0])
        output, root = pn_to_graph(fl)
        print(output.nodes)


if __name__ == "__main__":
    main()
