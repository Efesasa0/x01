import numpy as np
import torch

from x01.sr.blocks.unet import UNet

GOLDEN = np.load("tests/sr/golden/unet_golden.npz")

CFG = dict(
    in_channels=1,
    out_channels=1,
    latent_dims=32,
    channel_multipliers=(1, 2),
    num_res_blocks=1,
    attention_resolutions=(8,),
    image_resolution=16,
    dropout_rate=0.0,
    resample_with_conv=True,
)


def _load_golden_into(model: UNet) -> None:
    model_keys = set(model.state_dict().keys())
    golden_keys = set(GOLDEN.files) - {"x", "t", "out"}
    missing = model_keys - golden_keys
    extra = golden_keys - model_keys
    assert not missing, f"model expects keys not in golden: {sorted(missing)}"
    assert not extra, f"golden has keys not in model: {sorted(extra)}"
    sd = {name: torch.from_numpy(GOLDEN[name]) for name in model_keys}
    model.load_state_dict(sd, strict=True)


def test_unet_shape_no_condition_no_h():
    model = UNet(**CFG)
    x = torch.randn(1, 1, 16, 16)
    t = torch.randn(1)
    model.eval()
    with torch.no_grad():
        out = model(x, t)
    assert out.shape == (1, 1, 16, 16)


def test_unet_shape_with_condition_and_h():
    model = UNet(**CFG)
    x = torch.randn(1, 1, 16, 16)
    t = torch.randn(1)
    cond = torch.randn(1, 1, 16, 16)
    h = torch.randn(1)
    model.eval()
    with torch.no_grad():
        out = model(x, t, condition=cond, h=h)
    assert out.shape == (1, 1, 16, 16)


def test_unet_shape_resample_without_conv():
    # Covers the with_conv=False sampler path (broken in flax, see vendored-bugs memory).
    cfg = {**CFG, "resample_with_conv": False}
    model = UNet(**cfg)
    x = torch.randn(1, 1, 16, 16)
    t = torch.randn(1)
    model.eval()
    with torch.no_grad():
        out = model(x, t)
    assert out.shape == (1, 1, 16, 16)


def test_unet_golden():
    model = UNet(**CFG)
    _load_golden_into(model)

    x = torch.from_numpy(GOLDEN["x"])
    t = torch.from_numpy(GOLDEN["t"])
    expected = torch.from_numpy(GOLDEN["out"])

    model.eval()
    with torch.no_grad():
        out = model(x, t)

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
