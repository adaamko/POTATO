from collections import defaultdict
from typing import Dict, List
from typing import Union
from math import log2, sqrt
from rank_bm25 import BM25Okapi

import eli5
import pandas as pd
from xpotato.graph_extractor.extract import GraphExtractor, FeatureEvaluator
from xpotato.models.model import GraphModel
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import skcriteria as skc
from skcriteria.madm import simple
from skcriteria.pipeline import mkpipe
from skcriteria.preprocessing import invert_objectives, scalers


class GraphTrainer:
    def __init__(
        self,
        dataset: pd.DataFrame,
        lang: str = "en",
        max_edge: int = 2,
        max_features: int = 2000,
    ) -> None:
        print("Initializing trainer object...")
        self.dataset = dataset
        self.extractor = GraphExtractor(lang=lang, cache_fn="en_nlp_cache")
        self.evaluator = FeatureEvaluator()
        self.graph_model = GraphModel()
        self.max_edge = max_edge
        self.max_features = max_features
        self.model = LogisticRegression(random_state=0)

    def get_n_most_similar(
        self,
        sample: str,
        dataset: pd.DataFrame = None,
        n: int = 10,
        algorithm: str = "bm25",
    ) -> List[str]:
        corpus = self.dataset.text.tolist() if not dataset else dataset
        if algorithm == "bm25":
            tokenized_corpus = [doc.split() for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)

            query = sample.split()
            top_n = bm25.get_top_n(query, corpus, n=n)

            return top_n

        return corpus[0:10]

    def prepare_and_train(
        self, min_edge: int = 0, rank: bool = False
    ) -> Dict[str, List[List[Union[List[str], str]]]]:
        self.prepare()
        return self.train(min_edge=min_edge, rank=rank)

    def prepare(self) -> None:
        ids = pd.to_numeric(self.dataset.index).tolist()
        sentences = self.dataset.text.tolist()
        labels = self.dataset.label_id.tolist()
        graphs = self.dataset.graph.tolist()

        print(f"Featurizing graphs by generating subgraphs up to {self.max_edge}...")
        for ind, graph, label in tqdm(zip(ids, graphs, labels)):
            self.graph_model.featurize_sen_graph(ind, graph, label, self.max_edge)

        print("Getting feature graphs...")
        self.feature_graphs = self.graph_model.get_feature_graphs()
        self.feature_graph_strings = self.graph_model.get_feature_graph_strings()

        print("Selecting the best features...")
        if self.max_features > len(graphs):
            n_best = int(log2(len(graphs)) * sqrt(len(graphs)))
            self.graph_model.select_n_best(n_best)
        else:
            self.graph_model.select_n_best(self.max_features)

    def rank(
        self, features: Dict[str, List[List[Union[List[str], str]]]]
    ) -> Dict[str, List[List[Union[List[str], str]]]]:

        ranked_features = {}

        for key, val in features.items():
            stat, acc = self.evaluator.evaluate_feature(key, val, self.dataset)
            stat_opt = pd.DataFrame(
                {
                    "false_positives": stat.False_positive_sens.apply(lambda x: len(x)),
                    "true_positives": stat.True_positive_sens.apply(lambda x: len(x)),
                }
            )

            criteria_data = skc.mkdm(
                matrix=stat_opt,
                objectives=[min, max],
                criteria=stat_opt.columns,
                weights=[30, 70],
            )

            dm = mkpipe(
                invert_objectives.MinimizeToMaximize(),
                scalers.SumScaler(target="both"),
                simple.WeightedSumModel(),
            )

            dec = dm.evaluate(criteria_data)

            sorted_feats = [
                x for _, x in sorted(zip(dec.rank_, val), key=lambda pair: pair[0])
            ]
            ranked_features[key] = sorted_feats

        return ranked_features

    def train(
        self, min_edge: int = 1, rank: bool = False
    ) -> Dict[str, List[List[Union[List[str], str]]]]:
        label_vocab = {}
        for label in self.dataset.label.unique():
            label_vocab[label] = (
                self.dataset[self.dataset.label == label].iloc[0].label_id
            )

        inv_vocab = {v: k for k, v in label_vocab.items()}

        print("Generating training data...")
        train_X, train_Y = self.graph_model.get_x_y(
            self.dataset.label.tolist(), label_vocab=label_vocab
        )

        print("Training...")
        self.model.fit(train_X, train_Y)
        weights_df = eli5.explain_weights_df(self.model)
        features = defaultdict(list)

        print("Getting features...")
        for target in weights_df.target.unique():
            targeted_df = weights_df[weights_df.target == target]
            most_important_weights = []

            for i, w in enumerate(targeted_df.weight.tolist()):
                if w > 0.01:
                    most_important_weights.append(
                        targeted_df.iloc[i].feature.strip("x")
                    )

            for i in most_important_weights:
                if i != "<BIAS>":
                    g_nx = self.feature_graphs[self.graph_model.inverse_relabel[int(i)]]
                    g = self.feature_graph_strings[
                        self.graph_model.inverse_relabel[int(i)]
                    ]
                    if len(g_nx.edges()) >= min_edge:
                        features[inv_vocab[int(target)]].append(
                            ([g], [], inv_vocab[int(target)])
                        )

        if rank:
            print("Ranking features based on accuracy...")
            features = self.rank(features)

        return features
