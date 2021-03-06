import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class BaseG2NetFeatureEngineeringDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def _engineer_features(self, signals):
        raise NotImplementedError

    def __getitem__(self, index):
        path = self.paths[index]
        signal = np.load(path)
        features = self._engineer_features(signal)
        return features


def _collate_fn(batch):
    return batch


class G2NetFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, dataset, batch_size=256, num_workers=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def fit(self, X):
        return self

    def transform(self, paths):
        dataloader = torch.utils.data.DataLoader(
            self.dataset(paths),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )
        features = []
        for inputs in tqdm(dataloader):
            features.extend(inputs)
        return pd.DataFrame(features)
