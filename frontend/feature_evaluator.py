from sklearn.metrics import precision_recall_fscore_support
from tuw_nlp.graph.utils import GraphMatcher
import pandas as pd


def one_versus_rest(df, entity):
    mapper = {entity: 1}

    one_versus_rest_df = df.copy()
    one_versus_rest_df["one_versus_rest"] = [
        mapper[item] if item in mapper else 0 for item in df.label]

    return one_versus_rest_df


def evaluate_feature(cl, features, val_data):
    measure_features = []
    val_graphs = val_data.graph.tolist()
    val_labels = one_versus_rest(val_data, cl).one_versus_rest.tolist()

    for feat in features:
        measure = [feat[0]]
        matcher = GraphMatcher([feat])
        false_pos_g = []
        false_pos_s = []
        val_predicted = []
        for i, g in enumerate(val_graphs):
            feats = matcher.match(g)
            label = 0
            for feat in feats:
                label = 1
            if label == 1 and val_labels[i] == 0:
                false_pos_g.append(g)
                sen = val_data.iloc[i].sentence
                e1 = val_data.iloc[i].e1
                e2 = val_data.iloc[i].e2
                false_pos_s.append((sen, e1, e2))
            val_predicted.append(label)
        for pcf in precision_recall_fscore_support(val_labels, val_predicted, average=None):
            measure.append(pcf[1])
        measure.append(false_pos_g)
        measure.append(false_pos_s)

        measure_features.append(measure)
    df = pd.DataFrame(measure_features, columns=[
                      'Feature', 'Precision', 'Recall', "Fscore", "Support", "False_positive_graphs", "False_positive_sens"])

    return df
