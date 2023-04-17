import os

from xpotato.graph_extractor.rule import RuleSet


def get_features(path, label=None):
    files = []
    if os.path.isfile(path):
        assert path.endswith("json") or path.endswith(
            "tsv"
        ), "features file must be JSON or TSV"
        files.append(path)
    elif os.path.isdir(path):
        for fn in os.listdir(path):
            assert fn.endswith("json") or fn.endswith(
                "tsv"
            ), f"feature dir should only contain JSON or TSV files: {fn}"
            files.append(os.path.join(path, fn))
    else:
        raise ValueError(f"not a file or directory: {path}")

    labels = set()
    all_features = []
    for fn in files:
        ruleset = RuleSet()
        if fn.endswith("json"):
            ruleset.from_json(fn)
        elif fn.endswith("tsv"):
            ruleset.from_tsv(fn)
        else:
            raise ValueError(f"unknown file type: {fn}")
        features = ruleset.to_list()
        for k in features:
            lab = k[2]
            if label and label != lab:
                continue
            labels.add(lab)
            all_features.append(k)

    return all_features, labels
