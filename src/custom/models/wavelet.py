import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CWT(nn.Module):
    def __init__(self, widths=8, wavelet="ricker"):
        """PyTorch implementation of a continuous wavelet transform.
        Implemented using parts of SciPy and PyWavelets

        Ref: https://www.kaggle.com/anjum48/continuous-wavelet-transform-cwt-in-pytorch


        Args:
            widths (iterable): The wavelet scales to use, e.g. np.arange(1, 33)
            wavelet (str, optional): Name of wavelet. Either "ricker" or "morlet". Defaults to "ricker".
        """
        super().__init__()
        if isinstance(widths, int):
            widths = np.arange(1, widths - 1)

        self.widths = widths
        self.wavelet = getattr(self, wavelet)

    def ricker(self, points, a, dtype=None, device="cpu"):
        # https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/wavelets.py#L262-L306
        A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
        wsq = a ** 2
        vec = torch.arange(0, points, dtype=dtype).to(device) - (points - 1.0) / 2
        xsq = vec ** 2
        mod = 1 - xsq / wsq
        gauss = torch.exp(-xsq / (2 * wsq))
        total = A * mod * gauss
        return total

    def morlet(self, points, s, dtype=None, device="cpu"):
        x = torch.arange(0, points, dtype=dtype).to(device) - (points - 1.0) / 2
        x = x / s
        # https://uk.mathworks.com/help/wavelet/ref/morlet.html
        wavelet = torch.exp(-(x ** 2.0) / 2.0) * torch.cos(5.0 * x)
        output = np.sqrt(1 / s) * wavelet
        return output

    def forward(self, x):
        """Compute CWT arrays from a batch of multi-channel inputs

        Args:
            x (torch.tensor): Tensor of shape (batch_size, channels, time)

        Returns:
            torch.tensor: Tensor of shape (batch_size, channels, widths, time)
        """
        dtype = x.dtype
        device = x.device
        ch = x.shape[1]
        output = []

        for ind, width in enumerate(self.widths):
            N = np.min([10 * width, x.shape[-1]])
            wavelet_data = torch.conj(torch.flip(self.wavelet(N, width, dtype, device), [-1]))
            wavelet_data = torch.broadcast_to(wavelet_data, (ch, ch, wavelet_data.shape[0]))

            c = F.conv1d(x, wavelet_data, padding="same")
            output.append(c)

        return torch.stack(output, 2)
