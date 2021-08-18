import multiprocessing
import os
import warnings

import hydra
import nolds
import pandas as pd
from omegaconf import DictConfig

from base import BaseG2NetFeatureEngineeringDataset, G2NetFeatureEngineering


class NoldsFeatures(BaseG2NetFeatureEngineeringDataset):
    def _engineer_features(self, signals):
        features = {}
        for i, x in enumerate(signals):
            features[f"lyap_r_channel_{i}"] = nolds.lyap_r(x)
            for j, f in enumerate(nolds.lyap_e(x)):
                features[f"lyap_e_{j}_channel_{i}"] = f
            features[f"sampen_channel_{i}"] = nolds.sampen(x)
            features[f"hurst_rs_channel_{i}"] = nolds.hurst_rs(x)
            features[f"dfa_channel_{i}"] = nolds.dfa(x)

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
        NoldsFeatures, batch_size=num_workers, num_workers=num_workers
    )

    X_train = transformer.fit_transform(train["path"])
    X_test = transformer.transform(test["path"])

    print(X_train.info())

    X_train.to_pickle(os.path.join(features_dir, f"{filename}_train.pkl"))
    X_test.to_pickle(os.path.join(features_dir, f"{filename}_test.pkl"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
