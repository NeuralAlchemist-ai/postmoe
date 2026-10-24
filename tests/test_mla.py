import os
import pytest
import torch
import torch.nn as nn
from postmoe.config import Config
from postmoe.MLA import MLA_Attention
from pipeline import _make_linear


def test_rope_embeddings():
    cfg = Config(rope_dim=32, max_seq_len=128)
    mla_dummy = object.__new__(MLA_Attention)
    mla_dummy.config = cfg
    mla_dummy.rope_dim = cfg.rope_dim

    cos, sin = mla_dummy._get_rope_embeddings(None, 16, torch.device("cpu"), past_seq_len=0)
    assert cos.shape == (16, cfg.rope_dim)
    assert sin.shape == (16, cfg.rope_dim)

    cos_past, sin_past = mla_dummy._get_rope_embeddings(None, 8, torch.device("cpu"), past_seq_len=16)
    assert cos_past.shape == (8, cfg.rope_dim)
    assert sin_past.shape == (8, cfg.rope_dim)


def test_apply_rope_math():
    cfg = Config(rope_dim=4, max_seq_len=32)
    mla_dummy = object.__new__(MLA_Attention)
    mla_dummy.config = cfg

    cos = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    sin = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

    rotated = mla_dummy.apply_rope(x, cos, sin)
    assert torch.allclose(rotated, x), "Zero sine should keep vector unchanged."


def test_mla_attention_forward_cpu():
    hidden = 128
    n_head = 4
    nope_dim = 16
    rope_dim = 16
    latent_dim = 32
    head_dim = nope_dim
    max_seq_len = 64

    cfg = Config(
        n_head=n_head,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        kv_latent_dim=latent_dim,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
    )

    mla = MLA_Attention(
        config=cfg,
        kv_down_proj=_make_linear(torch.randn(latent_dim, hidden)),
        k_up_proj=_make_linear(torch.randn(n_head * nope_dim, latent_dim)),
        v_up_proj=_make_linear(torch.randn(n_head * head_dim, latent_dim)),
        q_nope=_make_linear(torch.randn(n_head * nope_dim, hidden)),
        q_rope=_make_linear(torch.randn(n_head * rope_dim, hidden)),
        k_rope=_make_linear(torch.randn(rope_dim, hidden)),
        o_proj=nn.Linear(n_head * head_dim, hidden, bias=False),
    )

    batch_size = 2
    seq_len = 8
    hidden_states = torch.randn(batch_size, seq_len, hidden)

    # Without cache
    out, cache = mla(hidden_states)
    assert out.shape == (batch_size, seq_len, hidden)
    assert cache is None

    # With use_cache=True
    out, (ckv_cache, kpe_cache) = mla(hidden_states, use_cache=True)
    assert out.shape == (batch_size, seq_len, hidden)
    assert ckv_cache.shape == (batch_size, seq_len, latent_dim)
    assert kpe_cache.shape == (batch_size, seq_len, 1, rope_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mla_attention_cuda_forward():
    hidden = 128
    n_head = 4
    nope_dim = 16
    rope_dim = 16
    latent_dim = 32
    head_dim = nope_dim
    max_seq_len = 64

    cfg = Config(
        n_head=n_head,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        kv_latent_dim=latent_dim,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
    )

    mla = MLA_Attention(
        config=cfg,
        kv_down_proj=_make_linear(torch.randn(latent_dim, hidden)),
        k_up_proj=_make_linear(torch.randn(n_head * nope_dim, latent_dim)),
        v_up_proj=_make_linear(torch.randn(n_head * head_dim, latent_dim)),
        q_nope=_make_linear(torch.randn(n_head * nope_dim, hidden)),
        q_rope=_make_linear(torch.randn(n_head * rope_dim, hidden)),
        k_rope=_make_linear(torch.randn(rope_dim, hidden)),
        o_proj=nn.Linear(n_head * head_dim, hidden, bias=False),
    ).cuda()

    batch_size = 2
    seq_len = 8
    hidden_states = torch.randn(batch_size, seq_len, hidden, device="cuda")

    # Without cache
    out, cache = mla(hidden_states)
    assert out.shape == (batch_size, seq_len, hidden)
    assert cache is None

    # With use_cache=True
    out, (ckv_cache, kpe_cache) = mla(hidden_states, use_cache=True)
    assert out.shape == (batch_size, seq_len, hidden)
    assert ckv_cache.shape == (batch_size, seq_len, latent_dim)
    assert kpe_cache.shape == (batch_size, seq_len, 1, rope_dim)


def test_vllm_architecture_converter(tmp_path):
    from postmoe.MLA import VLLMArchitectureConverter
    from types import SimpleNamespace

    hidden_size = 64
    n_head = 4
    num_kv_heads = 2
    orig_head_dim = hidden_size // n_head  # 16
    rope_dim = 8
    nope_dim = orig_head_dim - rope_dim    # 8
    kv_latent_dim = 16

    config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=n_head,
        num_key_value_heads=num_kv_heads,
        num_hidden_layers=2,
        vocab_size=1000,
        kv_latent_dim=kv_latent_dim,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        head_dim=orig_head_dim,
    )

    converter = VLLMArchitectureConverter(config)

    w_q = torch.randn(n_head * orig_head_dim, hidden_size)
    w_k = torch.randn(num_kv_heads * orig_head_dim, hidden_size)
    w_v = torch.randn(num_kv_heads * orig_head_dim, hidden_size)
    w_o = torch.randn(hidden_size, n_head * orig_head_dim)

    vllm_weights = converter.convert_attention_layer_to_vllm(
        w_q, w_k, w_v, w_o, split_strategy="suffix"
    )

    assert "q_proj.weight" in vllm_weights
    assert "q_rope_proj.weight" in vllm_weights
    assert "kv_down_proj.weight" in vllm_weights
    assert "k_up_proj.weight" in vllm_weights
    assert "v_up_proj.weight" in vllm_weights
    assert "k_rope_proj.weight" in vllm_weights
    assert "o_proj.weight" in vllm_weights

    assert vllm_weights["kv_down_proj.weight"].shape == (kv_latent_dim, hidden_size)
    assert vllm_weights["k_rope_proj.weight"].shape == (rope_dim, hidden_size)

    save_dir = str(tmp_path / "vllm_export")
    os.makedirs(save_dir, exist_ok=True)
    converter.generate_vllm_config(save_dir, num_experts=8, top_k=2)
    assert os.path.exists(os.path.join(save_dir, "config.json"))
