import argparse

from xpotato.graph_extractor.rule import RuleSet


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--features", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()

    features = args.features
    output = args.output

    rule_set = RuleSet()
    rule_set.from_json(features)

    rule_set.to_tsv(output)


if __name__ == "__main__":
    main()
