from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from xpotato.dataset.dataset import Dataset
from xpotato.dataset.explainable_sample import ExplainableSample
from xpotato.graph_extractor.graph import PotatoGraph


class ExplainableDataset(Dataset):
    def __init__(
        self,
        examples: List[Tuple[str, str]] = None,
        label_vocab: Dict[str, int] = {},
        lang="en",
        path=None,
        binary=False,
        cache_dir=None,
        cache_fn=None,
    ) -> None:
        super().__init__(
            examples=examples,
            label_vocab=label_vocab,
            lang=lang,
            path=path,
            binary=binary,
            cache_dir=cache_dir,
            cache_fn=cache_fn,
        )

    def read_dataset(
        self,
        examples: List[Tuple[str, str]] = None,
        path: str = None,
        binary: bool = False,
    ) -> List[ExplainableSample]:
        if examples:
            return [ExplainableSample(example) for example in examples]
        elif path:
            if binary:
                df = pd.read_pickle(path)
                graphs_str = self.prune_graphs(df.graph.tolist())
                df.drop(columns=["graph"], inplace=True)
                df["graph"] = graphs_str
            else:
                df = pd.read_csv(path, sep="\t")
            samples = [
                ExplainableSample(
                    (example["text"], example["label"], example["rationale"]),
                    potato_graph=PotatoGraph(graph_str=example["graph"]),
                    label_id=example["label_id"],
                )
                for _, example in tqdm(df.iterrows())
            ]
            self.graphs = [sample.potato_graph.graph for sample in samples]
            return samples
        else:
            raise ValueError("No examples or path provided")

    def to_dataframe(self, as_penman: bool = False) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "text": [sample.text for sample in self._dataset],
                "label": [sample.label for sample in self._dataset],
                "label_id": [
                    self.label_vocab[sample.label] if sample.label else None
                    for sample in self._dataset
                ],
                "rationale": [sample.rationale for sample in self._dataset],
                "graph": [
                    str(sample.potato_graph).replace("\n", " ")
                    if as_penman
                    else sample.potato_graph.graph
                    for sample in self._dataset
                ],
            }
        )
        return df
