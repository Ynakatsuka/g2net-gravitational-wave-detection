import os

import kvt
import numpy as np
import torch
from kvt.datasets import BaseDataset
from scipy import signal, stats


def whiten(signal):
    hann = torch.hann_window(len(signal), periodic=True, dtype=float)
    spec = torch.fft.fft(torch.from_numpy(signal).float() * hann)
    mag = torch.sqrt(torch.real(spec * torch.conj(spec)))

    return torch.real(torch.fft.ifft(spec / mag)).numpy() * np.sqrt(len(signal) / 2)


@kvt.DATASETS.register
class G2NetDataset(BaseDataset):
    def __init__(
        self,
        csv_filename,
        input_column,
        target_column=None,
        input_dir="../data/input",
        extension="",
        target_unique_values=None,
        enable_load=True,
        images_dir=None,
        split="train",
        transform=None,
        fold_column="Fold",
        num_fold=5,
        idx_fold=0,
        label_smoothing=0,
        return_input_as_x=True,
        # for pseudo labeling
        predictions_dirname_for_pseudo_labeling=None,
        test_csv_filename=None,
        test_images_dir=None,
        # for audio
        sample_rate=2048,
        normalize_mode=None,
        apply_bandpass=True,
        bandpass_lower_freq=20,
        bandpass_higher_freq=500,
        bandpass_filters=4,
        ignore_head_frames=0,
        apply_whiten=False,
        **params,
    ):
        super().__init__(
            csv_filename,
            input_column,
            target_column,
            input_dir,
            extension,
            target_unique_values,
            enable_load,
            images_dir,
            split,
            transform,
            fold_column,
            num_fold,
            idx_fold,
            label_smoothing,
            return_input_as_x,
            predictions_dirname_for_pseudo_labeling,
            test_csv_filename,
            test_images_dir,
            **params,
        )
        self.sample_rate = sample_rate
        self.normalize_mode = normalize_mode
        self.apply_bandpass = apply_bandpass
        self.ignore_head_frames = ignore_head_frames
        self.apply_whiten = apply_whiten

        Wn = (bandpass_lower_freq, bandpass_higher_freq)
        self.b, self.a = signal.butter(
            bandpass_filters, Wn, btype="bandpass", fs=sample_rate
        )

    def _load(self, path):
        x = np.load(path)
        return x

    def _extract_path_to_input_from_input_column(self, df):
        inputs = df[self.input_column].apply(
            lambda x: os.path.join(
                self.input_dir, self.images_dir, x[0], x[1], x[2], x + self.extension,
            )
        )
        if self.test_images_dir is not None:
            is_test = df["__is_test__"]
            test_inputs = df[self.input_column].apply(
                lambda x: os.path.join(
                    self.input_dir,
                    self.test_images_dir,
                    x[0],
                    x[1],
                    x[2],
                    x + self.extension,
                )
            )
            inputs[is_test] = test_inputs[is_test]
        return inputs.tolist()

    def _preprocess_input(self, x):
        """
        (max, min, mean, std)
        overall: 
            [4.6152116e-20 -4.4294355e-20 4.3110074e-26 6.1481726e-21]
        per channel:
            [4.6152116e-20 4.1438353e-20 1.1161064e-20]
            [-4.4294355e-20 -4.2303907e-20 -1.0863199e-20]
            [5.3645219e-27 1.2159634e-25 2.3708171e-27]
            [7.4209854e-21 7.4192858e-21 1.8380779e-21]
        target==0:
            [4.6152116e-20, -4.4294355e-20, 8.1091296e-26, 6.1511348e-21]
        target==1:
            [4.2472526e-20, -4.2402206e-20, 5.1104632e-27, 6.150679e-21]
        """
        if self.normalize_mode == 0:
            return x / x.max()
        elif self.normalize_mode == 1:
            return (x - x.mean()) / x.std()
        elif self.normalize_mode == 2:
            return (x - x.mean(axis=1).reshape(-1, 1)) / x.std(axis=1).reshape(-1, 1)
        elif self.normalize_mode == 3:
            return x / 4.6152116e-20
        elif self.normalize_mode == 4:
            return ((x + 4.4294355e-20) / 4.6152116e-20 - 0.5) * 2
        elif self.normalize_mode == 5:
            return (x - 4.3110074e-26) / 6.1481726e-21
        elif self.normalize_mode == 6:
            max_ = np.array([[4.6152116e-20], [4.1438353e-20], [1.1161064e-20]])
            return x / max_
        elif self.normalize_mode == 7:
            max_ = np.array([[4.6152116e-20], [4.1438353e-20], [1.1161064e-20]])
            min_ = np.array([[-4.4294355e-20], [-4.2303907e-20], [-1.0863199e-20]])
            return ((x - min_) / max_ - 0.5) * 2
        elif self.normalize_mode == 8:
            mean_ = np.array([[5.3645219e-27], [1.2159634e-25], [2.3708171e-27]])
            std_ = np.array([[7.4209854e-21], [7.4192858e-21], [1.8380779e-21]])
            return (x - mean_) / std_
        elif self.normalize_mode == 9:
            return (x - 5.1104632e-27) / 6.150679e-21
        elif self.normalize_mode == 10:
            return (x - 8.1091296e-26) / 6.1511348e-21
        elif self.normalize_mode == 11:
            x -= x.min() + 1e-07
            x /= x.max()
            x = np.concatenate(
                [np.expand_dims(stats.boxcox(x[i])[0], axis=0) for i in range(3)]
            )
            return x
        else:
            return (x - 4.3110074e-26) / 6.1481726e-21

    def apply_bandpass_filter(self, waves):
        return np.array([signal.filtfilt(self.b, self.a, wave) for wave in waves])

    def __getitem__(self, idx):
        if self.enable_load:
            path = self.inputs[idx]
            x = self._load(path)
        else:
            x = self.inputs[idx]

        x = self._preprocess_input(x)

        if self.apply_whiten:
            x = np.concatenate([np.expand_dims(whiten(x[i]), axis=0) for i in range(3)])

        if self.transform is not None:
            _x = x
            x = np.concatenate(
                [
                    np.expand_dims(self.transform(x[i], self.sample_rate), axis=0)
                    for i in range(3)
                ]
            )
            assert not np.array_equal(x, _x)

        if self.apply_bandpass:
            x = self.apply_bandpass_filter(x)

        if self.ignore_head_frames > 0:
            x = x[:, self.ignore_head_frames :]

        x = x.astype("float32")

        if self.return_input_as_x:
            inputs = {"x": x}
        else:
            inputs = x

        if self.targets is not None:
            inputs["y"] = self._preprocess_target(self.targets[idx])

        return inputs
