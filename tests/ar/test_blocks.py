import pytest
import torch

from x01.ar.blocks.dynamics import DynamicsBackBlock, DynamicsBlock, LowRankDynamicsBackBlock, LowRankDynamicsBlock
from x01.ar.blocks.embeddings import OverlapPatchEmbed
from x01.ar.blocks.samplers import Downsample, Upsample
from x01.ar.blocks.transformer import GatedDconvFeedForwardNetwork, MultiDConvHeadTransposedAttention, TransformerBlock

N = 16


# dynamics


def test_dynamics_init_is_orthogonal():
    K = DynamicsBlock(N).matrix.detach()
    torch.testing.assert_close(K.T @ K, torch.eye(N), atol=1e-5, rtol=0)


def test_dynamics_forward_is_right_matmul():
    block = DynamicsBlock(N)
    x = torch.randn(3, N)
    torch.testing.assert_close(block(x), x @ block.matrix)


def test_dynamics_preserves_norm_at_init():
    block = DynamicsBlock(N)
    x = torch.randn(3, N)
    torch.testing.assert_close(block(x).norm(dim=1), x.norm(dim=1), atol=1e-5, rtol=0)


def test_back_dynamics_separate_starts_as_copy():
    fwd = DynamicsBlock(N)
    back = DynamicsBackBlock(fwd, mode="separate")
    torch.testing.assert_close(back.matrix, fwd.matrix)
    assert back.matrix.data_ptr() != fwd.matrix.data_ptr()


def test_back_dynamics_single_is_transpose():
    fwd = DynamicsBlock(N)
    back = DynamicsBackBlock(fwd, mode="single")
    torch.testing.assert_close(back.matrix, fwd.matrix.T)
    x = torch.randn(2, N)
    torch.testing.assert_close(back(fwd(x)), x, atol=1e-5, rtol=0)


def test_back_dynamics_does_not_register_forward_params():
    fwd = DynamicsBlock(N)
    assert len(list(DynamicsBackBlock(fwd, mode="single").parameters())) == 0
    assert len(list(DynamicsBackBlock(fwd, mode="separate").parameters())) == 1


def test_back_dynamics_rejects_unknown_mode():
    with pytest.raises(ValueError):
        DynamicsBackBlock(DynamicsBlock(N), mode="both")


def test_low_rank_dynamics_shapes():
    fwd = LowRankDynamicsBlock(N, rank=4)
    back = LowRankDynamicsBackBlock(fwd, mode="separate")
    x = torch.randn(3, N)
    assert fwd.matrix.shape == (4, 4)
    assert fwd(x).shape == (3, N)
    assert back(x).shape == (3, N)


def test_low_rank_back_single_shares_projections():
    fwd = LowRankDynamicsBlock(N, rank=4)
    back = LowRankDynamicsBackBlock(fwd, mode="single")
    torch.testing.assert_close(back.matrix, fwd.core.T)
    assert len(list(back.parameters())) == 0


# embeddings and samplers


def test_overlap_patch_embed_shape():
    model = OverlapPatchEmbed(input_channels=3, embedding_dims=8, use_bias=False)
    assert model(torch.randn(2, 3, 16, 16)).shape == (2, 8, 16, 16)


@pytest.mark.parametrize("factor", [2, 4])
def test_downsample_upsample_shapes(factor):
    x = torch.randn(2, 16, 32, 32)
    down = Downsample(in_features=16, downsample_factor=factor)(x)
    assert down.shape == (2, 32, 32 // factor, 32 // factor)
    up = Upsample(in_features=32, upsample_factor=factor)(down)
    assert up.shape == (2, 16, 32, 32)


@pytest.mark.parametrize("cls", [Downsample, Upsample])
def test_sampler_rejects_unsupported_factor(cls):
    with pytest.raises(ValueError):
        cls(16, 3)


# transformer


def test_attention_shape():
    model = MultiDConvHeadTransposedAttention(dims=8, num_heads=2, use_bias=False)
    assert model(torch.randn(2, 8, 16, 16)).shape == (2, 8, 16, 16)


def test_gated_ffn_shape():
    model = GatedDconvFeedForwardNetwork(dims=8, ffn_expansion_factor=2.66, use_bias=False)
    assert model.hidden_dims == 21
    assert model(torch.randn(2, 8, 16, 16)).shape == (2, 8, 16, 16)


def test_transformer_block_shape_and_grad():
    model = TransformerBlock(dims=8, num_heads=2, ffn_expansion_factor=2, use_bias=False, use_norm_bias=True)
    x = torch.randn(2, 8, 16, 16, requires_grad=True)
    out = model(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
