import kvt
import kvt.losses
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@kvt.LOSSES.register
class SampleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        raise NotImplementedError
