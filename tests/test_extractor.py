import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from postmoe.extractor import UniversalAttentionDecoupler


class DummyAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_q_heads, num_kv_heads, head_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)


class DummyDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_q_heads, num_kv_heads, head_dim):
        super().__init__()
        self.self_attn = DummyAttentionLayer(hidden_size, num_q_heads, num_kv_heads, head_dim)


class DummyModel(nn.Module):
    def __init__(self, hidden_size, num_q_heads, num_kv_heads, head_dim, num_layers):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            DummyDecoderLayer(hidden_size, num_q_heads, num_kv_heads, head_dim)
            for _ in range(num_layers)
        ])


def create_dummy_setup(num_layers=2, hidden_size=64, num_q_heads=4, num_kv_heads=2, rope_dim=8):
    head_dim = hidden_size // num_q_heads  # 16
    config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        num_hidden_layers=num_layers,
    )
    model = DummyModel(hidden_size, num_q_heads, num_kv_heads, head_dim, num_layers)
    return model, config, rope_dim


def test_extractor_suffix_strategy():
    num_layers = 2
    hidden_size = 64
    num_q_heads = 4
    num_kv_heads = 2
    rope_dim = 8
    model, config, rope_dim = create_dummy_setup(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        rope_dim=rope_dim,
    )

    decoupler = UniversalAttentionDecoupler(model, config, rope_dim, split_strategy="suffix")
    W_k_nope, W_k_rope, W_q_nope, W_q_rope, W_v = decoupler.convert_to_target_layout(repeat_mode="gqa_broadcast")

    head_dim = hidden_size // num_q_heads  # 16
    nope_dim = head_dim - rope_dim         # 8

    # With gqa_broadcast, K heads are broadcast from num_kv_heads to num_q_heads
    assert W_k_nope.shape == (num_layers, num_q_heads, nope_dim, hidden_size)
    assert W_k_rope.shape == (num_layers, num_q_heads, rope_dim, hidden_size)
    assert W_q_nope.shape == (num_layers, num_q_heads, nope_dim, hidden_size)
    assert W_q_rope.shape == (num_layers, num_q_heads, rope_dim, hidden_size)
    assert W_v.shape == (num_layers, num_kv_heads, head_dim, hidden_size)


def test_extractor_repeat_mode_none():
    model, config, rope_dim = create_dummy_setup()
    decoupler = UniversalAttentionDecoupler(model, config, rope_dim, split_strategy="suffix")
    W_k_nope, W_k_rope, W_q_nope, W_q_rope, W_v = decoupler.convert_to_target_layout(repeat_mode=None)

    # Without broadcast, K retains num_kv_heads
    assert W_k_nope.shape[1] == config.num_key_value_heads
    assert W_k_rope.shape[1] == config.num_key_value_heads


def test_extractor_prefix_and_interleaved_strategies():
    model, config, rope_dim = create_dummy_setup()

    # Prefix strategy
    decoupler_prefix = UniversalAttentionDecoupler(model, config, rope_dim, split_strategy="prefix")
    res_prefix = decoupler_prefix.convert_to_target_layout(repeat_mode="gqa_broadcast")
    assert len(res_prefix) == 5

    # Interleaved strategy
    decoupler_inter = UniversalAttentionDecoupler(model, config, rope_dim, split_strategy="interleaved")
    res_inter = decoupler_inter.convert_to_target_layout(repeat_mode="gqa_broadcast")
    assert len(res_inter) == 5


def test_extractor_invalid_modes():
    model, config, rope_dim = create_dummy_setup()

    # Invalid split strategy
    decoupler = UniversalAttentionDecoupler(model, config, rope_dim, split_strategy="invalid_split")
    with pytest.raises(ValueError, match="Invalid split_strategy"):
        decoupler.convert_to_target_layout()

    # Invalid repeat mode
    decoupler_valid = UniversalAttentionDecoupler(model, config, rope_dim, split_strategy="suffix")
    with pytest.raises(ValueError, match="Invalid repeat_mode"):
        decoupler_valid.convert_to_target_layout(repeat_mode="unsupported_mode")
