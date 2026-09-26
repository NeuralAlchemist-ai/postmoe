import pytest
import torch
import torch.nn as nn
from postmoe.config import Config
from pipeline import _make_linear, Pipeline


def test_make_linear():
    weight = torch.randn(32, 64)
    lin = _make_linear(weight)

    assert isinstance(lin, nn.Linear), "Expected nn.Linear instance."
    assert lin.in_features == 64
    assert lin.out_features == 32
    assert lin.bias is None, "Expected bias=False."
    assert torch.allclose(lin.weight, weight)

    x = torch.randn(5, 64)
    out = lin(x)
    assert out.shape == (5, 32)
    assert torch.allclose(out, x @ weight.T)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLA_Attention requires CUDA")
def test_custom_mla_attention_construction():
    hidden = 64
    num_q_heads = 4
    num_kv_heads = 2
    nope_dim = 8
    rope_dim = 8
    head_dim = 16
    num_layers = 1

    config = Config(
        n_head=num_q_heads,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        head_dim=head_dim,
        max_seq_len=64,
    )
    pipeline = Pipeline(config)

    # Mock all_weights tuple:
    #   k_nope: [num_layers, num_q_heads, nope_dim, hidden]
    #   k_rope: [num_layers, num_q_heads, rope_dim, hidden]
    #   q_nope: [num_layers, num_q_heads, nope_dim, hidden]
    #   q_rope: [num_layers, num_q_heads, rope_dim, hidden]
    #   v:      [num_layers, num_kv_heads, head_dim, hidden]
    W_k_nope = torch.randn(num_layers, num_q_heads, nope_dim, hidden)
    W_k_rope = torch.randn(num_layers, num_q_heads, rope_dim, hidden)
    W_q_nope = torch.randn(num_layers, num_q_heads, nope_dim, hidden)
    W_q_rope = torch.randn(num_layers, num_q_heads, rope_dim, hidden)
    W_v = torch.randn(num_layers, num_kv_heads, head_dim, hidden)
    pipeline.all_weights = (W_k_nope, W_k_rope, W_q_nope, W_q_rope, W_v)

    o_proj = nn.Linear(num_q_heads * head_dim, hidden, bias=False)
    mla_module = pipeline.custom_MLA_attention(0, o_proj)

    assert mla_module is not None
    assert mla_module.config.n_head == num_q_heads
    assert mla_module.config.nope_dim == nope_dim
    assert mla_module.config.rope_dim == rope_dim
    assert mla_module.o_proj is o_proj
    assert mla_module.kv_down_proj.weight.shape[1] == hidden
    assert mla_module.k_up_proj.weight.shape[0] == num_q_heads * nope_dim
    assert mla_module.v_up_proj.weight.shape[0] == num_q_heads * head_dim
