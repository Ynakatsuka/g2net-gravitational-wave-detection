import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def sklearn_accuracy(
    pred, target, threshold=0.5,
):
    pred = pred > threshold
    score = accuracy_score(target, pred)
    return score


def sklearn_roc_auc_score(
    pred, target,
):
    score = roc_auc_score(target, pred)
    return score


def sklearn_precision_score(pred, target, threshold=0.5):
    score = precision_score(target, pred >= threshold)
    return score


def sklearn_recall_score(pred, target, threshold=0.5):
    score = recall_score(target, pred >= threshold)
    return score

