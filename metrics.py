import numpy as np
from sklearn.metrics import cluster
from scipy.optimize import linear_sum_assignment
from munkres import Munkres
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    #ind = linear_assignment(w.max()-w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def pairwise_precision(y_true, y_pred):
    true_positives, false_positives, _, _ = _pairwise_confusion(y_true, y_pred)
    return true_positives / (true_positives + false_positives)

def pairwise_recall(y_true, y_pred):
    true_positives, _, false_negatives, _ = _pairwise_confusion(y_true, y_pred)
    return true_positives / (true_positives + false_negatives)

def _pairwise_confusion(y_true, y_pred):
    contingency = cluster.contingency_matrix(y_true, y_pred)
    same_class_true = np.max(contingency, 1)
    same_class_pred = np.max(contingency, 0)
    diff_class_true = contingency.sum(axis=1) - same_class_true
    diff_class_pred = contingency.sum(axis=0) - same_class_pred
    total = contingency.sum()

    true_positives = (same_class_true * (same_class_true - 1)).sum()
    false_positives = (diff_class_true * same_class_true * 2).sum()
    false_negatives = (diff_class_pred * same_class_pred * 2).sum()
    true_negatives = total * (
      total - 1) - true_positives - false_positives - false_negatives

    return true_positives, false_positives, false_negatives, true_negatives

def f_score(true_label, pred_label):
    # best mapping between true_label and predict label
    l1 = list(set(true_label))
    numclass1 = len(l1)

    l2 = list(set(pred_label))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        # print('Class Not equal, Error!!!!')
        # return 0
        precision = pairwise_precision(true_label, pred_label)
        recall = pairwise_recall(true_label, pred_label)
        F1 = 2 * precision * recall / (precision + recall)
        return F1, F1

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]
        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        new_predict[ai] = c
    f1_macro = metrics.f1_score(true_label, new_predict, average='macro')
    precision_macro = metrics.precision_score(true_label, new_predict, average='macro')
    recall_macro = metrics.recall_score(true_label, new_predict, average='macro')
    f1_micro = metrics.f1_score(true_label, new_predict, average='micro')
    precision_micro = metrics.precision_score(true_label, new_predict, average='micro')
    recall_micro = metrics.recall_score(true_label, new_predict, average='micro')
    return f1_macro, f1_micro

def eva_metrics(true, pred):
    acc = accuracy(true, pred)
    nmi = normalized_mutual_info_score(true, pred)
    f1, _ = f_score(true, pred)
    ari = adjusted_rand_score(true, pred)
    acc = round(acc, 4)
    nmi = round(nmi, 4)
    f1 = round(f1, 4)
    ari = round(ari, 4)
    return [acc, nmi, f1, ari]



