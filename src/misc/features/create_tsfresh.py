import multiprocessing
import os
import warnings

import hydra
import pandas as pd
import tsfresh
from omegaconf import DictConfig

from base import BaseG2NetFeatureEngineeringDataset, G2NetFeatureEngineering


class TsFreshFeatures(BaseG2NetFeatureEngineeringDataset):
    def _engineer_features(self, signals):
        df = pd.DataFrame(signals.T, columns=["channel_0", "channel_1", "channel_2"])
        df["id"] = 0
        extracted_features = tsfresh.extract_features(
            df, column_id="id", n_jobs=0, disable_progressbar=True
        )
        features = {}
        for k, v in extracted_features.items():
            features[k] = v.values[0]

        return features


@hydra.main(config_path="../../../config", config_name="default")
def main(config: DictConfig) -> None:
    filename = __file__.split("/")[-1][:-3]
    input_dir = config.input_dir
    features_dir = config.features_dir
    os.makedirs(features_dir, exist_ok=True)

    train = pd.read_csv(config.competition.train_path)
    test = pd.read_csv(config.competition.test_path)

    train["path"] = train["id"].apply(
        lambda x: f"{input_dir}/train/{x[0]}/{x[1]}/{x[2]}/{x}.npy"
    )
    test["path"] = test["id"].apply(
        lambda x: f"{input_dir}/test/{x[0]}/{x[1]}/{x[2]}/{x}.npy"
    )

    num_workers = multiprocessing.cpu_count()
    transformer = G2NetFeatureEngineering(
        TsFreshFeatures, batch_size=num_workers, num_workers=num_workers
    )

    X_train = transformer.fit_transform(train["path"])
    X_test = transformer.transform(test["path"])

    print(X_train.info())

    X_train.to_pickle(os.path.join(features_dir, f"{filename}_train.pkl"))
    X_test.to_pickle(os.path.join(features_dir, f"{filename}_test.pkl"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
