from sklearn.metrics import precision_recall_fscore_support
from tuw_nlp.graph.utils import GraphFormulaMatcher
import pandas as pd


def one_versus_rest(df, entity):
    mapper = {entity: 1}

    one_versus_rest_df = df.copy()
    one_versus_rest_df["one_versus_rest"] = [
        mapper[item] if item in mapper else 0 for item in df.label]

    return one_versus_rest_df


def evaluate_feature(cl, features, data):
    measure_features = []
    graphs = data.graph.tolist()
    labels = one_versus_rest(data, cl).one_versus_rest.tolist()

    for feat in features:
        measure = [feat[0]]
        matcher = GraphFormulaMatcher([feat])
        false_pos_g = []
        false_pos_s = []
        true_pos_g = []
        true_pos_s = []
        false_neg_g = []
        false_neg_s = []
        predicted = []
        for i, g in enumerate(graphs):
            feats = matcher.match(g)
            label = 0
            for feat in feats:
                label = 1
            if label == 1 and labels[i] == 0:
                false_pos_g.append(g)
                sen = data.iloc[i].sentence
                e1 = data.iloc[i].e1
                e2 = data.iloc[i].e2
                lab = data.iloc[i].label
                false_pos_s.append((sen, e1, e2, lab))
            if label == 1 and labels[i] == 1:
                true_pos_g.append(g)
                sen = data.iloc[i].sentence
                e1 = data.iloc[i].e1
                e2 = data.iloc[i].e2
                lab = data.iloc[i].label
                true_pos_s.append((sen, e1, e2, lab))
            if label == 0 and labels[i] == 1:
                false_neg_g.append(g)
                sen = data.iloc[i].sentence
                e1 = data.iloc[i].e1
                e2 = data.iloc[i].e2
                lab = data.iloc[i].label
                false_neg_s.append((sen, e1, e2, lab))
            predicted.append(label)
        for pcf in precision_recall_fscore_support(labels, predicted, average=None):
            measure.append(pcf[1])
        measure.append(false_pos_g)
        measure.append(false_pos_s)
        measure.append(true_pos_g)
        measure.append(true_pos_s)
        measure.append(false_neg_g)
        measure.append(false_neg_s)
        measure_features.append(measure)

    predicted = []
    matcher = GraphFormulaMatcher(features)
    for i, g in enumerate(graphs):
        feats = matcher.match(g)
        label = 0
        for feat in feats:
            label = 1
        predicted.append(label)

    accuracy = []
    for pcf in precision_recall_fscore_support(labels, predicted, average=None):
        accuracy.append(pcf[1])

    df = pd.DataFrame(measure_features, columns=[
                      'Feature', 'Precision', 'Recall', "Fscore", "Support", "False_positive_graphs", "False_positive_sens", "True_positive_graphs", "True_positive_sens", "False_negative_graphs", "False_negative_sens"])

    return df, accuracy
