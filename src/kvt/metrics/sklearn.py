import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def sklearn_accuracy(
    pred,
    target,
    threshold=0.5,
):
    pred = pred > threshold
    score = accuracy_score(target, pred)
    return score


def sklearn_roc_auc_score(
    pred,
    target,
):
    score = roc_auc_score(target, pred)
    return score
