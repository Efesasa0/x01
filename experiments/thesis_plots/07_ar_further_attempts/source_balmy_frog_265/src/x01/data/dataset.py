from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class KMFlowDataset(Dataset):
    SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}

    @classmethod
    def split_bounds(cls, n_traj: int, split: str) -> tuple[int, int]:
        assert split in cls.SPLIT_FRACTIONS, f"split must be train/val/test, got {split}"
        n_train = int(n_traj * cls.SPLIT_FRACTIONS["train"])
        n_val = int(n_traj * cls.SPLIT_FRACTIONS["val"])
        bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, n_traj),
        }
        return bounds[split]

    def __init__(
        self,
        path: str,
        split: str = "train",
        mode: Literal["pairs", "frames"] = "pairs",
        downsample: int = 4,
        std: float = None,
        skip_n_frames: int = 0,
        pick_n_trajs: int = None,
    ):
        """
        mode="pairs"  -> __getitem__ returns (x_t, x_{t+1}); loses the last frame per traj. For AR training.
        mode="frames" -> __getitem__ returns x_t alone; keeps every frame. For SR / diffusion training.
        """
        data = np.load(path, mmap_mode="r")  # (N_traj, T, H, W)
        i0, i1 = self.split_bounds(data.shape[0], split)
        if pick_n_trajs is not None:
            i1 = min(i1, i0 + pick_n_trajs)
        data = np.array(data[i0:i1])  # (n_split, T, H, W)
        if skip_n_frames > 0:
            data = data[:, skip_n_frames:]
        data = data[:, :, ::downsample, ::downsample]
        data = data.astype(np.float32)

        if std is not None:
            self.std = float(std)
            data = data / self.std
        else:
            self.std = None
        self.mode = mode
        N_traj, _T_full, H, W = data.shape
        self.n_trajs = N_traj

        if mode == "pairs":
            x = data[:, :-1]  # (N_traj, T-1, H, W)
            y = data[:, 1:]  # (N_traj, T-1, H, W)
            T = x.shape[1]
            self.samples_per_traj = T
            self.x = torch.tensor(x.reshape(N_traj * T, H, W)).unsqueeze(1)
            self.y = torch.tensor(y.reshape(N_traj * T, H, W)).unsqueeze(1)
        elif mode == "frames":
            T = data.shape[1]
            self.samples_per_traj = T
            self.x = torch.tensor(data.reshape(N_traj * T, H, W)).unsqueeze(1)
            self.y = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.mode == "pairs":
            return self.x[idx], self.y[idx]
        if self.mode == "frames":
            return self.x[idx]
