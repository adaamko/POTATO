from typing import Dict, List, Tuple

import pandas as pd
from xpotato.dataset.dataset import Dataset
from xpotato.dataset.relation_sample import RelationSample


class RelationDataset(Dataset):
    def __init__(
        self, examples: List[Tuple[str, str]], label_vocab: Dict[str, int], lang="en"
    ) -> None:
        super().__init__(examples=examples, label_vocab=label_vocab, lang=lang)

    def read_dataset(self, examples: List[Tuple[str, str]]) -> List[RelationSample]:
        return [RelationSample(example) for example in examples]

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "text": [sample.text for sample in self._dataset],
                "label": [sample.label for sample in self._dataset],
                "label_id": [
                    self.label_vocab[sample.label] if sample.label else None
                    for sample in self._dataset
                ],
                "entity1": [sample.entity1 for sample in self._dataset],
                "entity2": [sample.entity2 for sample in self._dataset],
                "graph": [sample.graph for sample in self._dataset],
            }
        )
        return df
