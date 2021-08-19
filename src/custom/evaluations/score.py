import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import kvt


@kvt.METRICS.register
def hard_sample_roc_auc_score(pred, target, threshold=0.01):
    paths = sorted(glob.glob("../data/output/predictions/oof/default/*.npy"))

    fold = pd.read_csv("../data/input/train_fold_v000.csv")
    fold["pred"] = 0
    for i, path in enumerate(paths):
        fold.loc[fold.Fold == i, "pred"] = np.load(path)
        if np.mean((fold.loc[fold.Fold == i, "target"].values - target) ** 2) <= 0.0001:
            idx = fold.Fold == i
    fold["deviation"] = (fold["target"] - fold["pred"]) ** 2
    idx &= fold["deviation"] > threshold
    assert sum(idx) > 10000

    score = roc_auc_score(target[idx], pred[idx])
    return score
