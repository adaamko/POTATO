from typing import List, Tuple

import pandas as pd
import stanza
from exprel.dataset.sample import Sample


class Dataset:
    def __init__(self, examples: List[Tuple[str, str]], lang="en") -> None:
        self.nlp = stanza.Pipeline(lang)
        self._dataset = self.read_dataset(examples)

    def read_dataset(self, examples: List[Tuple[str, str]]) -> List[Sample]:
        return [Sample(example) for example in examples]

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({"text": [sample.text for sample in self._dataset], "label": [
                          sample.label for sample in self._dataset]})
        return df
