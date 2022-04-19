from typing import Dict, List, Tuple

import pandas as pd
from xpotato.dataset.dataset import Dataset
from xpotato.dataset.explainable_sample import ExplainableSample


class ExplainableDataset(Dataset):
    def __init__(
        self, examples: List[Tuple[str, str]], label_vocab: Dict[str, int], lang="en"
    ) -> None:
        super().__init__(examples=examples, label_vocab=label_vocab, lang=lang)

    def read_dataset(self, examples: List[Tuple[str, str]]) -> List[ExplainableSample]:
        return [ExplainableSample(example) for example in examples]

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
