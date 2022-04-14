import numpy as np
from collections import defaultdict
from tuw_nlp.common.vocabulary import Vocabulary
from tuw_nlp.graph.lexical import LexGraphs
from tuw_nlp.graph.utils import graph_to_pn
from xpotato.models.utils import tree_to_code


class GraphModel:
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
        self.random_state = 1234

    def get_feature_graph_strings(self):
        return [graph_to_pn(G) for G in self.get_feature_graphs()]

    def get_feature_graphs(self):
        return [self.lexgraphs.from_tuple(T) for T in self.get_feature_names()]

    def get_feature_names(self):
        return [self.feature_vocab.get_word(i) for i in range(len(self.feature_vocab))]

    def convert_tree_to_features(self, clf, feature_graph_strings):
        features = defaultdict(list)

        for j, est in enumerate(clf.estimators_):
            paths = [
                i
                for i in list(
                    tree_to_code(est, feature_graph_strings, self.inverse_relabel)
                )
                if i[2]
            ]
            for path in paths:
                features[list(self.label_vocab.word_to_id.keys())[j]].append(
                    (path[0], path[1], self.label_vocab.id_to_word[j])
                )

        return features

    def featurize_sen_graph(self, sen_id, graph, attr, max_edge=1):
        feats = set()
        self.sen_ids.append(sen_id)
        for sg_tuple, sg in self.lexgraphs.gen_lex_subgraphs(graph, max_edge):
            feats.add(self.feature_vocab.get_id(sg_tuple, allow_new=True))

        self.feats_by_sen[sen_id] = feats

        self.labels[sen_id] = attr
        self.vocab_size = len(self.feature_vocab)

    def select_n_best(self, max_features):
        relabel_dict, feature_num = self.feature_vocab.select_n_best(max_features)
        self.vocab_size = feature_num
        self.relabel_dict = relabel_dict
        self.inverse_relabel = {relabel_dict[k]: k for k in relabel_dict}

    def select_n_best_from_each_class(self, max_features, feature_graphs):
        edge_to_ind = defaultdict(list)
        for i, graph in enumerate(feature_graphs):
            edge_to_ind[len(graph.edges(data=True))].append(i)

        relabel_dict, feature_num = self.feature_vocab.select_n_best_from_each_class(
            max_features, edge_to_ind, up_to=3
        )
        self.vocab_size = feature_num
        self.relabel_dict = relabel_dict
        self.inverse_relabel = {relabel_dict[k]: k for k in relabel_dict}

    def get_x_y(self, attr, label_vocab=None):
        X = np.zeros((len(self.sen_ids), self.vocab_size))
        y = np.zeros(len(self.sen_ids))
        for i, sen_id in enumerate(self.sen_ids):
            for j in self.feats_by_sen[sen_id]:
                if self.relabel_dict:
                    if j in self.relabel_dict:
                        X[i][self.relabel_dict[j]] = 1
                else:
                    X[i][j] = 1
            y[i] = (
                label_vocab[attr[i]]
                if label_vocab
                else self.label_vocab.get_id(attr[i], allow_new=True)
            )

        return X, y
