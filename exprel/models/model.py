import argparse
import json
import logging
import os
import sys

import numpy as np
from tuw_nlp.common.vocabulary import Vocabulary
from tuw_nlp.graph.lexical import LexGraphs
from tuw_nlp.graph.utils import graph_to_pn
from tuw_nlp.text.utils import load_parsed, save_parsed


class GraphModel():
    def __init__(self):
        self.lexgraphs = LexGraphs()
        self.feature_vocab = Vocabulary()
        self.label_vocab = Vocabulary()
        self.labels = {}
        self.feats_by_sen = {}
        self.sen_ids = []
        self.vocab_size = 0
        self.relabel_dict = {}
        self.inverse_relabel = {}

    def get_feature_graph_strings(self):
        return [graph_to_pn(G) for G in self.get_feature_graphs()]

    def get_feature_graphs(self):
        return [
            self.lexgraphs.from_tuple(T) for T in self.get_feature_names()]

    def get_feature_names(self):
        return [
            self.feature_vocab.get_word(i) for i in range(
                len(self.feature_vocab))]

    def featurize_sen_graph(self, sen_id, graph, attr, max_edge=1):
        feats = set()
        self.sen_ids.append(sen_id)
        for sg_tuple, sg in self.lexgraphs.gen_lex_subgraphs(graph, max_edge):
            feats.add(self.feature_vocab.get_id(sg_tuple))

        self.feats_by_sen[sen_id] = feats

        self.labels[sen_id] = attr
        self.vocab_size = len(self.feature_vocab)
    
    def select_n_best(self, max_features):
        relabel_dict, feature_num = self.feature_vocab.select_n_best(max_features)
        self.vocab_size = feature_num
        self.relabel_dict = relabel_dict
        self.inverse_relabel = {relabel_dict[k] : k for k in relabel_dict}
            
    def get_x_y(self, attr):
        X = np.zeros((len(self.sen_ids), self.vocab_size))
        y = np.zeros(len(self.sen_ids))
        for i, sen_id in enumerate(self.sen_ids):
            for j in self.feats_by_sen[sen_id]:
                if self.relabel_dict:
                    if j in self.relabel_dict:
                        X[i][self.relabel_dict[j]] = 1
                else:
                    X[i][j] = 1
            y[i] = self.label_vocab.get_id(attr[i])

        return X, y
