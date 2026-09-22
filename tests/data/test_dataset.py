import numpy as np
import pytest
import torch

from x01.data.dataset import KMFlowDataset, load_full_trajectory

N_TRAJ, N_FRAMES, RES = 6, 12, 16


@pytest.fixture
def data_path(tmp_path):
    # frame value encodes (trajectory, time) so indexing is checkable
    traj = np.arange(N_TRAJ).reshape(-1, 1, 1, 1) * 100
    time = np.arange(N_FRAMES).reshape(1, -1, 1, 1)
    data = np.broadcast_to(traj + time, (N_TRAJ, N_FRAMES, RES, RES)).astype(np.float32)
    path = tmp_path / "kmflow.npy"
    np.save(path, data)
    return str(path)


def make(data_path, **overrides):
    kwargs = dict(split="train", n_train=3, n_val=2, n_test=1, T_in=2, T=8, downsample=4)
    return KMFlowDataset(data_path, **{**kwargs, **overrides})


def test_split_bounds_are_contiguous_and_disjoint():
    bounds = [KMFlowDataset.split_bounds(s, 3, 2, 1, 6) for s in ("train", "val", "test")]
    assert bounds == [(0, 3), (3, 5), (5, 6)]


def test_split_bounds_rejects_oversized_split():
    with pytest.raises(AssertionError):
        KMFlowDataset.split_bounds("train", 4, 2, 1, 6)


def test_pairs_shapes(data_path):
    ds = make(data_path, n_rollout=3)
    assert len(ds) == 3 * (8 - 3)
    x, y = ds[0]
    assert x.shape == (1, 4, 4)
    assert y.shape == (3, 1, 4, 4)


def test_pairs_targets_are_next_frames(data_path):
    ds = make(data_path, split="val", n_rollout=2)
    x, y = ds[1]  # trajectory 3, sample 1 -> absolute frame T_in + 1
    assert x[0, 0, 0].item() == 303
    assert [y[k, 0, 0, 0].item() for k in range(2)] == [304, 305]


def test_frames_mode(data_path):
    ds = make(data_path, split="test", mode="frames")
    assert ds.y is None
    assert len(ds) == 8
    assert ds[0].shape == (1, 4, 4)
    assert ds[0][0, 0, 0].item() == 502


def test_std_scaling(data_path):
    ds = make(data_path, std=2.0)
    assert ds.std == 2.0
    assert ds[0][0][0, 0, 0].item() == 1.0


def test_window_must_fit_trajectory(data_path):
    with pytest.raises(AssertionError):
        make(data_path, T_in=5, T=10)


def test_load_full_trajectory(data_path):
    frames = load_full_trajectory(np.load(data_path), idx=1, T=5, std=10.0, downsample=2)
    assert frames.shape == (5, 1, 8, 8)
    assert frames.dtype == torch.float32
    torch.testing.assert_close(frames[:, 0, 0, 0], torch.tensor([10.0, 10.1, 10.2, 10.3, 10.4]))
