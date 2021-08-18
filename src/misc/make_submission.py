import glob
import os
import pprint
import sys

sys.path.append("src/")

import hydra
import kvt
import numpy as np
import pandas as pd
from kvt.utils import update_experiment_name
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../../config", config_name="default")
def main(config: DictConfig) -> None:
    # fix experiment name
    config = update_experiment_name(config)

    print("-" * 100)
    pprint.PrettyPrinter(indent=2).pprint(OmegaConf.to_container(config, resolve=True))

    # variables
    sample_submission_path = config.competition.sample_submission_path
    target_column = config.competition.target_column

    # load
    sub = pd.read_csv(sample_submission_path)
    load_oof_paths = sorted(glob.glob(f"{config.trainer.inference.dirpath}/*.npy"))
    assert len(load_oof_paths) > 0
    if len(load_oof_paths) == 1:
        preds = np.load(load_oof_paths[0])
    else:
        preds = np.mean([np.load(path) for path in load_oof_paths], axis=0)

    # postprocess
    sub[target_column] = preds

    # print stats
    print("[prediction describe]")
    print(sub[target_column].describe())

    # save submission
    dirpath = os.path.join(config.save_dir, "submission")
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    sub.to_csv(os.path.join(dirpath, f"{config.experiment_name}.csv"), index=False)


if __name__ == "__main__":
    main()
