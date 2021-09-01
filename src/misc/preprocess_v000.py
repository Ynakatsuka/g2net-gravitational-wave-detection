import os
import sys

import custom
import hydra
import kvt
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from hydra.utils import instantiate
from kvt.builder import build_dataloaders
from kvt.initialization import initialize
from nnAudio.Spectrogram import CQT1992v2
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

# local libraries
sys.path.append("src/")


def save_images(dataloader, save_dir, is_train):
    if is_train:
        save_dir = os.path.join(save_dir, "train")
    else:
        save_dir = os.path.join(save_dir, "test")
    os.makedirs(save_dir, exist_ok=True)

    spectrogram_extractor = CQT1992v2(
        sr=2048, fmin=10, fmax=1024, hop_length=16, bins_per_octave=8,
    )

    ids = dataloader.dataset.inputs
    ids = [_id.split("/")[-1] for _id in ids]

    pivot = 0
    for inputs in tqdm(dataloader):
        x = inputs["x"]
        images = []
        for i in range(x.shape[1]):
            images.append(spectrogram_extractor(x[:, i, :]).unsqueeze(1))
        images = torch.cat(images, dim=1)

        for image in images:
            filename = ids[pivot]
            save_path = os.path.join(save_dir, filename)
            np.save(save_path, image)
            pivot += 1


@hydra.main(config_path="../../config", config_name="default")
def main(config: DictConfig) -> None:
    initialize()

    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.dataset.dataset[0].params.fold_column = None
        config.dataset.dataset[1].params.fold_column = None

    config = edict(OmegaConf.to_container(config, resolve=True))

    filename = __file__.split("/")[-1][:-3]
    save_dir = os.path.join(config.save_dir, filename)

    # build dataloaders
    dataloaders = build_dataloaders(config, drop_last=False, shuffle=False)

    save_images(dataloaders["train"], save_dir, is_train=True)
    save_images(dataloaders["test"], save_dir, is_train=False)


if __name__ == "__main__":
    main()
