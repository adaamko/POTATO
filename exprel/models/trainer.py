import eli5
import pandas as pd
from exprel.feature_extractor.extract import FeatureExtractor
from exprel.models.model import GraphModel
from tqdm import tqdm
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as split


class GraphTrainer():
    def __init__(self, dataset: pd.DataFrame, lang: str = "en", max_edge: int = 2, max_features: int = 2000) -> None:
        print("Initializing trainer object...")
        self.dataset = dataset
        self.extractor = FeatureExtractor(lang=lang, cache_fn="en_nlp_cache")
        self.model = GraphModel()
        self.max_edge = max_edge
        self.max_features = max_features
        self.model = LogisticRegression(random_state=0)

    def train() -> None:
        ids = pd.to_numeric(self.dataset.index).tolist()
        sentences = self.dataset.text.tolist()
        labels = self.dataset.label_id.tolist()
        graphs = self.dataset.graph.tolist()

        print(
            f"Featurizing graphs by generating subgraphs up to {max_edge}...")
        for ind, graph, label in tqdm(zip(ids, postprocessed_graphs, labels)):
            model.featurize_sen_graph(ind, graph, label, 2)

        print("Getting feature graphs...")
        self.feature_graphs = self.model.get_feature_graphs()
        self.feature_graph_strings = self.model.get_feature_graph_strings()

        print("Selecting the best features...")
        model.select_n_best(self.max_features)

        label_vocab = {}
        for label in self.dataset.label.unique():
            label_vocab[label] = self.dataset[self.dataset.label ==
                                              label].iloc[0].label_id

        print("Generating training data...")
        train_X, train_Y = model.get_x_y(
            self.dataset.label, label_vocab=label_vocab)

        tr_data, val_data, tr_labels, val_labels = split(
            train_X, train_Y, test_size=0.2, random_state=1234)

        weights_df = eli5.explain_weights_df(clf)
        features = defaultdict(list)

        for target in weights_df.target.unique():
            targeted_df = weights_df[weights_df.target == target]
            most_important_weights = targeted_df.iloc[:5].feature.str.strip(
                "x").tolist()
            for i in most_important_weights:
                if i != "<BIAS>":
                    g_nx = feature_graphs[model.inverse_relabel[int(i)]]
                    g = feature_graph_strings[model.inverse_relabel[int(i)]]
                    features[list(keys)[int(target)]].append(
                        ([g], [], {v: k for k, v in label_vocab.items()}[int(target)]))


        return features, weights_df