import numpy as np
import torch

from ar.blocks.transformer import GatedDconvFeedForwardNetwork, MultiDConvHeadTransposedAttention, TransformerBlock

GOLDEN = np.load("tests/ar/golden/mdha_golden.npz")
GOLDEN_GDFFN = np.load("tests/ar/golden/gdffn_golden.npz")
GOLDEN_TB = np.load("tests/ar/golden/transformer_block_golden.npz")


def test_mdha_shape():
    model = MultiDConvHeadTransposedAttention(dims=8, num_heads=2, use_bias=False)
    x = torch.randn(1, 8, 16, 16)
    assert model(x).shape == (1, 8, 16, 16)


def test_mdha_golden():
    W_qkv = torch.tensor(GOLDEN["W_qkv"]).permute(3, 2, 0, 1)
    W_down = torch.tensor(GOLDEN["W_down"]).permute(3, 2, 0, 1)
    W_out = torch.tensor(GOLDEN["W_out"]).permute(3, 2, 0, 1)
    temp = torch.tensor(GOLDEN["temp"])
    x = torch.tensor(GOLDEN["x"]).permute(0, 3, 1, 2)
    expected = torch.tensor(GOLDEN["out"]).permute(0, 3, 1, 2)

    model = MultiDConvHeadTransposedAttention(dims=8, num_heads=2, use_bias=False)
    model.qkv.weight = torch.nn.Parameter(W_qkv)
    model.qkv_downconv.weight = torch.nn.Parameter(W_down)
    model.out_projection.weight = torch.nn.Parameter(W_out)
    model.temperature = torch.nn.Parameter(temp)

    err = (model(x) - expected).abs().max().item()
    assert err < 1e-4, f"mdha golden error {err:.2e}"


def test_gdffn_shape():
    model = GatedDconvFeedForwardNetwork(dims=8, ffn_expansion_factor=2, use_bias=False)
    x = torch.randn(1, 8, 16, 16)
    assert model(x).shape == (1, 8, 16, 16)


def test_gdffn_golden():
    W_in = torch.tensor(GOLDEN_GDFFN["W_in"]).permute(3, 2, 0, 1)
    W_down = torch.tensor(GOLDEN_GDFFN["W_down"]).permute(3, 2, 0, 1)
    W_out = torch.tensor(GOLDEN_GDFFN["W_out"]).permute(3, 2, 0, 1)
    x = torch.tensor(GOLDEN_GDFFN["x"]).permute(0, 3, 1, 2)
    expected = torch.tensor(GOLDEN_GDFFN["out"]).permute(0, 3, 1, 2)

    model = GatedDconvFeedForwardNetwork(dims=8, ffn_expansion_factor=2, use_bias=False)
    model.input_projection.weight = torch.nn.Parameter(W_in)
    model.down_conv.weight = torch.nn.Parameter(W_down)
    model.out_projection.weight = torch.nn.Parameter(W_out)

    err = (model(x) - expected).abs().max().item()
    assert err < 2e-4, f"gdffn golden error {err:.2e}"


def test_transformer_block_shape():
    model = TransformerBlock(dims=8, num_heads=2, ffn_expansion_factor=2, use_bias=False, use_norm_bias=True)
    x = torch.randn(1, 8, 16, 16)
    assert model(x).shape == (1, 8, 16, 16)


def test_transformer_block_golden():
    g = GOLDEN_TB
    model = TransformerBlock(dims=8, num_heads=2, ffn_expansion_factor=2, use_bias=False, use_norm_bias=True)

    model.norm1.weight = torch.nn.Parameter(torch.tensor(g["norm1_scale"]))
    model.norm1.bias = torch.nn.Parameter(torch.tensor(g["norm1_bias"]))
    model.norm2.weight = torch.nn.Parameter(torch.tensor(g["norm2_scale"]))
    model.norm2.bias = torch.nn.Parameter(torch.tensor(g["norm2_bias"]))

    model.attn.qkv.weight = torch.nn.Parameter(torch.tensor(g["W_qkv"]).permute(3, 2, 0, 1))
    model.attn.qkv_downconv.weight = torch.nn.Parameter(torch.tensor(g["W_down"]).permute(3, 2, 0, 1))
    model.attn.out_projection.weight = torch.nn.Parameter(torch.tensor(g["W_out"]).permute(3, 2, 0, 1))
    model.attn.temperature = torch.nn.Parameter(torch.tensor(g["temp"]))

    model.ffn.input_projection.weight = torch.nn.Parameter(torch.tensor(g["W_ffn_in"]).permute(3, 2, 0, 1))
    model.ffn.down_conv.weight = torch.nn.Parameter(torch.tensor(g["W_ffn_down"]).permute(3, 2, 0, 1))
    model.ffn.out_projection.weight = torch.nn.Parameter(torch.tensor(g["W_ffn_out"]).permute(3, 2, 0, 1))

    x = torch.tensor(g["x"]).permute(0, 3, 1, 2)
    expected = torch.tensor(g["out"]).permute(0, 3, 1, 2)

    err = (model(x) - expected).abs().max().item()
    assert err < 2e-4, f"transformer block golden error {err:.2e}"
