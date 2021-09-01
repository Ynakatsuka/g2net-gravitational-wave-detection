import math
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import psutil
import torch
from omegaconf import OmegaConf, open_dict


def seed_torch(seed=None, random_seed=True):
    if random_seed or seed is None:
        seed = np.random.randint(0, 1000000)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


@contextmanager
def trace(title, logger=None):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2.0 ** 30
    yield
    m1 = p.memory_info()[0] / 2.0 ** 30
    delta = m1 - m0
    sign = "+" if delta >= 0 else "-"
    delta = math.fabs(delta)
    message = (
        f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} "
    )
    print(message)
    if logger is not None:
        logger.info(message)


def check_attr(config, name):
    if hasattr(config, name):
        return config[name]
    else:
        return False


def update_experiment_name(config):
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.experiment_name = ",".join(
            [
                e
                for e in config.experiment_name.split(",")
                if ("trainer.idx_fold" not in e) and ("run=" not in e)
            ]
        )
        if not config.experiment_name:
            config.experiment_name = "default"

        if hasattr(config.trainer, "logger") and (
            not config.trainer.logger.name
        ):
            config.trainer.logger.name = "default"

    return config


def concatenate(results):
    if len(results) == 0:
        return results

    if isinstance(results[0], np.ndarray):
        return np.concatenate(results, axis=0)
    elif isinstance(results[0], torch.Tensor):
        return torch.vstack([r.detach() for r in results])
    else:
        raise ValueError(f"Invalid result type: {type(results[0])}")


def save_predictions(predictions, dirpath, filename, split="validation"):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    path = os.path.join(dirpath, f"{split}_{filename}",)
    np.save(path, predictions)
