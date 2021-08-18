import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .base import LightningModuleBase


def disable_bn(model):
    for module in model.modules():
        if (
            isinstance(module, nn.BatchNorm1d)
            or isinstance(module, nn.BatchNorm2d)
            or isinstance(module, nn.BatchNorm3d)
            or isinstance(module, nn.SyncBatchNorm)
        ):
            module.eval()


def enable_bn(model):
    model.train()


class LightningModuleSAM(LightningModuleBase):
    automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x, aux_x, y, aux_y = self.extract_inputs_from_batch(batch)

        if self.train_transform is not None:
            x = self.train_transform(x)

        optimizer = self.optimizers()

        loss = self._calculate_loss(x, y, aux_x, aux_y)
        self.manual_backward(loss)
        optimizer.first_step(zero_grad=True)

        disable_bn(self.model)
        _loss = self._calculate_loss(x, y, aux_x, aux_y)
        self.manual_backward(_loss)
        optimizer.second_step(zero_grad=True)
        enable_bn(self.model)

        return {"loss": loss}
