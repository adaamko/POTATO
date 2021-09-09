import eli5
import pandas as pd
from exprel.graph_extractor.extract import GraphExtractor
from exprel.models.model import GraphModel
from tqdm import tqdm
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as split


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
        self.graph_model = GraphModel()
        self.max_edge = max_edge
        self.max_features = max_features
        self.model = LogisticRegression(random_state=0)

    def train(self) -> None:
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
        self.graph_model.select_n_best(self.max_features)

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
                    features[inv_vocab[int(target)]].append(
                        ([g], [], inv_vocab[int(target)])
                    )

        return features
