import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class KMFlowDataset(Dataset):
    """
    Kolmogorov flow dataset for AR/Koopman training.

    Loads high-resolution vorticity fields, uniformly downsamples to
    low-resolution, and returns consecutive frame pairs (x_t, x_{t+1}).

    Parameters
    ----------
    path : str
        Path to kmflow_highres.npy, shape (40, 320, 256, 256).
    split : {'train', 'val', 'test'}, optional, default='train'
        Dataset split.
    downsample : int, optional, default=4
        Stride for uniform spatial downsampling (4 gives 256 to 64).
    normalize : bool, optional, default=False
        If True, divide by std. Mean is 0 by construction for this dataset.
    std : float, optional, default=None
        Required when normalize=True. Always compute from the train split first.
    """

    SPLIT_INDICES = {
        "train": (0, 32),
        "val": (32, 36),
        "test": (36, 40),
    }

    def __init__(
        self, path: str, split: str = "train", downsample: int = 4, normalize: bool = False, std: float = None
    ):
        assert split in self.SPLIT_INDICES, f"split must be train/val/test, got {split}"

        data = np.load(path)  # (40, 320, 256, 256)
        i0, i1 = self.SPLIT_INDICES[split]
        data = data[i0:i1]  # (N_traj, 320, 256, 256)
        data = data[:, :, ::downsample, ::downsample]
        data = data.astype(np.float32)

        if normalize:
            if std is None:
                raise ValueError(
                    "std must be provided when normalize=True. Compute it from the train split and pass it in."
                )
            self.std = float(std)
            data = data / self.std
        else:
            self.std = None

        # build consecutive pairs across all trajectories
        # each trajectory has 319 pairs (t, t+1)
        x = data[:, :-1]  # (N_traj, 319, H, W)
        y = data[:, 1:]  # (N_traj, 319, H, W)

        N_traj, T, H, W = x.shape
        x = x.reshape(N_traj * T, H, W)
        y = y.reshape(N_traj * T, H, W)

        # add channel dim NCHW (N, C, H, W)
        self.x = torch.tensor(x).unsqueeze(1)  # (N_pairs, 1, 64, 64)
        self.y = torch.tensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == "__main__":
    path = "../kmflow_highres.npy"

    # without normalization
    for split in ("train", "val", "test"):
        ds = KMFlowDataset(path, split=split)
        x0, _ = ds[0]
        print(f"{split:5s}  samples={len(ds)}  shape={tuple(x0.shape)}  min={x0.min():.3f}  max={x0.max():.3f}")

    # with normalization — std from train only
    train_std = float(np.load(path)[:32].std())
    print(f"\ntrain std: {train_std:.4f}")
    for split in ("train", "val", "test"):
        ds = KMFlowDataset(path, split=split, normalize=True, std=train_std)
        x0, _ = ds[0]
        print(f"{split:5s}  samples={len(ds)}  min={x0.min():.3f}  max={x0.max():.3f}")

    # error case
    try:
        KMFlowDataset(path, split="val", normalize=True)
    except ValueError as e:
        print(f"\nExpected error: {e}")
