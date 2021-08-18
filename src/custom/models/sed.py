from functools import partial

import kvt
import kvt.models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kvt.augmentation import SpecAugmentationPlusPlus
from kvt.models.layers import AttBlockV2
from kvt.models.sound_event_detections import (
    Loudness,
    PCENTransform,
    add_frequency_encoding,
    add_time_encoding,
    make_delta,
)
from nnAudio.Spectrogram import CQT1992v2, CQT2010v2
from torchlibrosa.augmentation import SpecAugmentation
from torchvision import transforms

from .wavelet import CWT


def gem(x, kernel_size, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), kernel_size).pow(1.0 / p)


def do_mixup(x, lam, indices):
    shuffled_x = x[indices]
    x = lam * x + (1 - lam) * shuffled_x
    return x


@kvt.MODELS.register
class G2Net(nn.Module):
    def __init__(
        self,
        backbone,
        spectrogram_method="CQT",
        sample_rate=2048,
        hop_length=32,
        fmin=10,
        fmax=1024,
        bins_per_octave=8,
        trainable=False,
        norm=False,
        window="hann",
        widths=8,
        use_spec_augmentation=False,
        time_drop_width=64,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2,
        spec_augmentation_method=None,
        apply_tta=False,
        apply_mixup=False,
        apply_spec_shuffle=False,
        spec_shuffle_prob=0,
        normalize_specs=False,
        apply_spectral_centroid=False,
        apply_pcen=False,
        apply_delta_spectrum=False,
        apply_time_freq_encoding=False,
        resize_shape=None,
        **params,
    ):
        super().__init__()
        self.apply_tta = apply_tta
        self.apply_mixup = apply_mixup
        self.apply_spec_shuffle = apply_spec_shuffle
        self.spec_shuffle_prob = spec_shuffle_prob
        self.normalize_specs = normalize_specs
        self.apply_spectral_centroid = apply_spectral_centroid
        self.apply_pcen = apply_pcen
        self.apply_delta_spectrum = apply_delta_spectrum
        self.apply_time_freq_encoding = apply_time_freq_encoding
        self.resize_shape = resize_shape

        if isinstance(window, list):
            window = tuple(window)

        # Spectrogram extractor
        if spectrogram_method == "CQT":
            self.spectrogram_extractor = CQT1992v2(
                sr=sample_rate,
                fmin=fmin,
                fmax=fmax,
                hop_length=hop_length,
                bins_per_octave=bins_per_octave,
                trainable=trainable,
                norm=norm,
                window=window,
            )
        elif spectrogram_method == "STFT":
            func = partial(
                torch.stft,
                n_fft=fmax,
                hop_length=hop_length,
                normalized=norm,
                return_complex=True,
            )
            self.spectrogram_extractor = lambda x: func(x).abs()
        elif spectrogram_method == "CWT":
            self.spectrogram_extractor = CWT(widths=widths)

        else:
            raise ValueError

        # Spec augmenter
        self.spec_augmenter = None
        if use_spec_augmentation and (spec_augmentation_method is None):
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=time_drop_width,
                time_stripes_num=time_stripes_num,
                freq_drop_width=freq_drop_width,
                freq_stripes_num=freq_stripes_num,
            )
        elif use_spec_augmentation and (spec_augmentation_method is not None):
            self.spec_augmenter = SpecAugmentationPlusPlus(
                time_drop_width=time_drop_width,
                time_stripes_num=time_stripes_num,
                freq_drop_width=freq_drop_width,
                freq_stripes_num=freq_stripes_num,
                method=spec_augmentation_method,
            )

        if self.apply_spectral_centroid:
            self.spectral_centroid_bn = nn.BatchNorm1d(1)

        if self.apply_pcen:
            self.pcen_transform = PCENTransform(trainable=False)

        if self.resize_shape is not None:
            self.resizer = transforms.Resize(resize_shape)

        # layers = list(backbone.children())[:-2]
        # self.backbone = nn.Sequential(*layers)
        self.backbone = backbone

    def forward(
        self, input, mixup_lambda=None, mixup_index=None, return_spectrogram=False
    ):
        # (batch_size, 3, time_steps) -> (batch_size, 3, freq_bins, time_steps)
        x = []
        for i in range(input.shape[1]):
            x.append(self.spectrogram_extractor(input[:, i, :]).unsqueeze(1))
        x = torch.cat(x, dim=1)

        if return_spectrogram:
            return x

        if self.normalize_specs:
            m = x.view(x.size()[0], -1).mean()
            s = x.view(x.size()[0], -1).std()
            x = (x - m) / s

        # augmentation
        # (batch_size, 3, freq_bins, time_steps) -> (batch_size, 3, freq_bins, time_steps)
        if (
            self.training
            and self.apply_spec_shuffle
            and (np.random.rand() < self.spec_shuffle_prob)
        ):
            # (batch_size, 3, time_steps, freq_bins)
            idx = torch.randperm(x.shape[3])
            x = x[:, :, :, idx]

        if (self.training or self.apply_tta) and (self.spec_augmenter is not None):
            x = self.spec_augmenter(x)

        # additional features
        additional_features = []
        if self.apply_spectral_centroid:
            spectral_centroid = x.mean(-1)
            spectral_centroid = self.spectral_centroid_bn(spectral_centroid)
            spectral_centroid = spectral_centroid.unsqueeze(-1)
            spectral_centroid = spectral_centroid.repeat(1, 1, 1, self.n_mels)
            additional_features.append(spectral_centroid)

        if self.apply_delta_spectrum:
            delta_1 = make_delta(x)
            delta_2 = make_delta(delta_1)
            additional_features.extend([delta_1, delta_2])

        if self.apply_time_freq_encoding:
            freq_encode = add_frequency_encoding(x)
            time_encode = add_time_encoding(x)
            additional_features.extend([freq_encode, time_encode])

        if self.apply_pcen:
            pcen = self.pcen_transform(x)
            additional_features.append(pcen)

        if len(additional_features) > 0:
            additional_features.append(x)
            x = torch.cat(additional_features, dim=1)

        # Resize
        if self.resize_shape is not None:
            x = self.resizer(x)

        # Mixup on spectrogram
        # (batch_size, 3, freq_bins, time_steps) -> (batch_size, 3, freq_bins, time_steps)
        if self.training and self.apply_mixup and (mixup_lambda is not None):
            x = do_mixup(x, mixup_lambda, mixup_index)

        # (batch_size, 3, freq_bins, time_steps) -> (batch_size, 1)
        output = self.backbone(x)

        return output


@kvt.MODELS.register
class AttentionG2Net(G2Net):
    def __init__(
        self,
        backbone,
        spectrogram_method="CQT",
        sample_rate=2048,
        hop_length=32,
        fmin=10,
        fmax=1024,
        bins_per_octave=8,
        trainable=False,
        norm=False,
        window="hann",
        use_spec_augmentation=False,
        time_drop_width=64,
        time_stripes_num=2,
        freq_drop_width=8,
        freq_stripes_num=2,
        spec_augmentation_method=None,
        apply_tta=False,
        apply_mixup=False,
        apply_spec_shuffle=False,
        spec_shuffle_prob=0,
        in_features=1280,
        num_classes=1,
        use_gru_layer=False,
        dropout_rate=0.0,
        pooling_kernel_size=3,
        **params,
    ):
        super().__init__(
            backbone=backbone,
            spectrogram_method=spectrogram_method,
            sample_rate=sample_rate,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            bins_per_octave=bins_per_octave,
            trainable=trainable,
            norm=norm,
            window=window,
            use_spec_augmentation=use_spec_augmentation,
            time_drop_width=time_drop_width,
            time_stripes_num=time_stripes_num,
            freq_drop_width=freq_drop_width,
            freq_stripes_num=freq_stripes_num,
            spec_augmentation_method=spec_augmentation_method,
            apply_tta=apply_tta,
            apply_mixup=apply_mixup,
            apply_spec_shuffle=apply_spec_shuffle,
            spec_shuffle_prob=spec_shuffle_prob,
            **params,
        )
        self.use_gru_layer = use_gru_layer
        self.dropout_rate = dropout_rate
        self.pooling_kernel_size = pooling_kernel_size

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, num_classes, activation="sigmoid")

        if self.use_gru_layer:
            self.gru = nn.GRU(in_features, in_features, batch_first=True)

    def forward(
        self, input, mixup_lambda=None, mixup_index=None, return_spectrogram=False
    ):
        # (batch_size, 3, time_steps) -> (batch_size, 3, freq_bins, time_steps)
        x = []
        for i in range(input.shape[1]):
            x.append(self.spectrogram_extractor(input[:, i, :]).unsqueeze(1))
        x = torch.cat(x, dim=1)

        if return_spectrogram:
            return x

        # augmentation
        # (batch_size, 3, freq_bins, time_steps) -> (batch_size, 3, freq_bins, time_steps)
        if (
            self.training
            and self.apply_spec_shuffle
            and (np.random.rand() < self.spec_shuffle_prob)
        ):
            # (batch_size, 3, time_steps, freq_bins)
            idx = torch.randperm(x.shape[3])
            x = x[:, :, :, idx]

        if (self.training or self.apply_tta) and (self.spec_augmenter is not None):
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        # (batch_size, 3, freq_bins, time_steps) -> (batch_size, 3, freq_bins, time_steps)
        if self.training and self.apply_mixup and (mixup_lambda is not None):
            x = do_mixup(x, mixup_lambda, mixup_index)

        # (batch_size, 3, freq_bins, time_steps) -> (batch_size, channels, freq, frames)
        x = self.backbone(x)

        # (batch_size, channels, freq, frames) -> (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # GRU
        if self.use_gru_layer:
            # (batch_size, channels, frames) -> (batch_size, channels, frames)
            x = x.transpose(1, 2).contiguous()
            x, _ = self.gru(x)
            x = x.transpose(1, 2).contiguous()

        # channel smoothing
        # (batch_size, channels, frames) -> (batch_size, channels, frames)
        x = gem(x, kernel_size=self.pooling_kernel_size)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # (batch_size, channels, frames) -> (batch_size, channels, frames)
        x = x.transpose(1, 2).contiguous()
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2).contiguous()

        # (batch_size, channels, frames) -> (batch_size, 1)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        output = torch.sum(norm_att * self.att_block.cla(x), dim=2)

        return output
