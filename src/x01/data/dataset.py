from typing import Literal

import numpy as np
import torch
from physicsnemo.distributed.manager import DistributedManager
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class KMFlowDataset(Dataset):
    @staticmethod
    def split_bounds(split: str, n_train: int, n_val: int, n_test: int, n_traj: int) -> tuple[int, int]:
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split}"
        total = n_train + n_val + n_test
        assert total <= n_traj, f"n_train+n_val+n_test={total} exceeds dataset size {n_traj}"
        bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, n_train + n_val + n_test),
        }
        return bounds[split]

    def __init__(
        self,
        path: str,
        split: str,
        n_train: int,
        n_val: int,
        n_test: int,
        T_in: int,
        T: int,
        mode: Literal["pairs", "frames"] = "pairs",
        downsample: int = 4,
        std: float = None,
        n_rollout: int = 1,
    ):
        data = np.load(path, mmap_mode="r")  # (N_traj, T_full, H, W)
        i0, i1 = self.split_bounds(split, n_train, n_val, n_test, data.shape[0])
        assert T_in + T <= data.shape[1], f"T_in+T={T_in + T} exceeds trajectory length {data.shape[1]}"
        data = np.array(data[i0:i1, T_in : T_in + T])  # (n_split, T, H, W)
        data = data[:, :, ::downsample, ::downsample]
        data = data.astype(np.float32)

        if std is not None:
            self.std = float(std)
            data = data / self.std
        else:
            self.std = None
        self.mode = mode
        self.n_rollout = int(n_rollout)
        assert self.n_rollout >= 1, f"n_rollout must be >= 1, got {self.n_rollout}"
        N_traj, T_data, H, W = data.shape
        self.n_trajs = N_traj

        if mode == "pairs":
            assert T_data > self.n_rollout, f"trajectory length {T_data} must exceed n_rollout {self.n_rollout}"
            n_samples = T_data - self.n_rollout
            self.samples_per_traj = n_samples
            x = data[:, :n_samples]  # (N_traj, n_samples, H, W)
            # y_seq[:, i, k, :, :] = frame at time i + k + 1
            y_seq = np.stack(
                [data[:, k + 1 : k + 1 + n_samples] for k in range(self.n_rollout)], axis=2
            )  # (N_traj, n_samples, n_rollout, H, W)
            self.x = torch.tensor(x.reshape(N_traj * n_samples, H, W)).unsqueeze(1)
            self.y = torch.tensor(y_seq.reshape(N_traj * n_samples, self.n_rollout, H, W)).unsqueeze(2)
            # self.y: (N_traj * n_samples, n_rollout, 1, H, W)
        elif mode == "frames":
            n_frames = data.shape[1]
            self.samples_per_traj = n_frames
            self.x = torch.tensor(data.reshape(N_traj * n_frames, H, W)).unsqueeze(1)
            self.y = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.mode == "pairs":
            return self.x[idx], self.y[idx]
        if self.mode == "frames":
            return self.x[idx]


def load_full_trajectory(data, idx: int, T: int, std: float = 1.0, downsample: int = 4) -> torch.Tensor:
    frames = np.array(data[idx, :T, ::downsample, ::downsample], dtype=np.float32)
    frames = frames / std
    return torch.from_numpy(frames).unsqueeze(1)


def make_loader(
    data_path: str,
    split: str,
    std: float,
    batch_size: int,
    dist: DistributedManager,
    n_train: int,
    n_val: int,
    n_test: int,
    T_in: int,
    T: int,
    num_workers: int = 4,
    mode: Literal["pairs", "frames"] = "pairs",
    downsample: int = 4,
    n_rollout: int = 1,
) -> tuple[DataLoader, DistributedSampler | None]:
    ds = KMFlowDataset(
        data_path,
        split=split,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        T_in=T_in,
        T=T,
        mode=mode,
        std=std,
        downsample=downsample,
        n_rollout=n_rollout,
    )
    sampler = None
    if dist.distributed and split == "train":
        sampler = DistributedSampler(ds, num_replicas=dist.world_size, rank=dist.rank, shuffle=True)
    return (
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train" and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train" and sampler is not None),  # Drops last uneven batch
        ),
        sampler,
    )
