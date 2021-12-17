from xpotato.graph_extractor.extract import GraphExtractor
from xpotato.graph_extractor.extract import FeatureEvaluator
from xpotato.dataset.utils import default_pn_to_graph
from tuw_nlp.graph.utils import (
    GraphFormulaMatcher,
)

import json
import os

from fastapi import FastAPI
from pydantic import BaseModel


FEATURE_PATH = os.getenv("FEATURE_PATH", None)
GRAPH_FORMAT = os.getenv("GRAPH_FORMAT", None)
LANG = os.getenv("LANG", None)

assert FEATURE_PATH, "FEATURE_PATH is not set"
assert GRAPH_FORMAT, "GRAPH_FORMAT is not set"
assert LANG, "LANG is not set"


def match_texts(text):
    texts = text.split("\n")

    graphs = list(EXTRACTOR.parse_iterable([text for text in texts], GRAPH_FORMAT))

    predicted = []

    for i, g in enumerate(graphs):
        feats = MATCHER.match(g)
        label = "NONE"
        for key, feature in feats:
            label = key
        predicted.append(label)

    return predicted


class Item(BaseModel):
    text: str


app = FastAPI()


@app.on_event("startup")
def init_data():
    global EXTRACTOR
    global FEATURES
    global EVALUATOR
    global MATCHER
    EXTRACTOR = GraphExtractor(lang=LANG, cache_fn=f"{LANG}_nlp_cache")

    if GRAPH_FORMAT == "ud":
        EXTRACTOR.init_nlp()
    elif GRAPH_FORMAT == "amr":
        EXTRACTOR.init_amr()

    with open(FEATURE_PATH) as f:
        FEATURES = json.load(f)

    EVALUATOR = FeatureEvaluator()

    feature_values = []
    for k in FEATURES:
        for f in FEATURES[k]:
            feature_values.append(f)
    MATCHER = GraphFormulaMatcher(feature_values, converter=default_pn_to_graph)


@app.post("/")
async def infer(item: Item):
    predicted = match_texts(text=item.text)
    return predicted


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8000, reload=True)
