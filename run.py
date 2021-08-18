import os
import pprint
import sys
import warnings

import hydra
import pytorch_lightning as pl
from easydict import EasyDict as edict
from omegaconf import DictConfig, OmegaConf

# local libraries
sys.path.append("src/")
import custom  # import all custom modules for registering objects.
from kvt.apis.evaluate_oof import run as run_evaluate_oof
from kvt.apis.inference import run as run_inference
from kvt.apis.train import run as run_train
from kvt.initialization import initialize
from kvt.utils import update_experiment_name


def train(config):
    print("-" * 100)
    print("train")
    run_train(config)


def inference(config):
    print("-" * 100)
    print("inference")
    run_inference(config)


def evaluate_oof(config):
    print("-" * 100)
    print("evaluate_oof")
    run_evaluate_oof(config)


@hydra.main(config_path="config", config_name="default")
def main(config: DictConfig) -> None:
    # fix experiment name
    config = update_experiment_name(config)

    # convert to easydict
    config = edict(OmegaConf.to_container(config, resolve=True))

    if hasattr(config, "numexpr_max_threads"):
        os.environ["NUMEXPR_MAX_THREADS"] = config.numexpr_max_threads

    if config.disable_warnings:
        warnings.filterwarnings("ignore")

    if config.print_config:
        pprint.PrettyPrinter(indent=2).pprint(config)

    if config.run in ("train", "inference", "evaluate_oof"):
        # initialize torch
        pl.seed_everything(config.seed, workers=True)

        # run main function
        eval(config.run)(config)
    else:
        raise ValueError(f"Invalid run mode: {config.run}.")


if __name__ == "__main__":
    initialize()
    main()
