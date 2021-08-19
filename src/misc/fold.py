import os
import pprint
import sys

sys.path.append("src/")

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import kvt


@hydra.main(config_path="../../config", config_name="default")
def main(config: DictConfig) -> None:
    print("-" * 100)
    pprint.PrettyPrinter(indent=2).pprint(OmegaConf.to_container(config, resolve=True))

    fold_column = config.fold.fold_column
    train = pd.read_csv(config.competition.train_path)
    y = train[config.competition.target_column]
    groups = None
    if hasattr(config.competition, "group_column") and (
        config.competition.group_column is not None
    ):
        groups = train[config.competition.group_column]

    # split
    train[fold_column] = 0
    kfold = instantiate(config.fold.fold)
    for f, (_, valid_index) in enumerate(kfold.split(train, y=y, groups=groups)):
        train.loc[valid_index, fold_column] = f
    path = os.path.join(config.input_dir, config.fold.csv_filename)
    train.to_csv(path, index=False)


if __name__ == "__main__":
    main()
