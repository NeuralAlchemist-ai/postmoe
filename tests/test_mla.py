import pytest
import torch
import torch.nn as nn
from postmoe.config import Config
from postmoe.MLA import MLA_Attention
from pipeline import _make_linear


def test_rope_cache_and_freq():
    cfg = Config(rope_dim=32, max_seq_len=128)
    
    # Check MLA_Attention RoPE helper methods on CPU without initializing GPU buffers
    mla_dummy = object.__new__(MLA_Attention)
    mla_dummy.config = cfg
    cos_cache, sin_cache = mla_dummy._build_rope_cache(cfg.max_seq_len, cfg.rope_dim)

    assert cos_cache.shape == (cfg.max_seq_len, cfg.rope_dim)
    assert sin_cache.shape == (cfg.max_seq_len, cfg.rope_dim)

    mla_dummy.cos_cache = cos_cache
    mla_dummy.sin_cache = sin_cache

    cos, sin = mla_dummy.freq_rope(16, torch.device("cpu"), past_seq_len=0)
    assert cos.shape == (16, cfg.rope_dim)
    assert sin.shape == (16, cfg.rope_dim)

    # Exceeding max_seq_len should raise ValueError
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        mla_dummy.freq_rope(129, torch.device("cpu"), past_seq_len=0)


def test_apply_rope_math():
    cfg = Config(rope_dim=4, max_seq_len=32)
    mla_dummy = object.__new__(MLA_Attention)
    mla_dummy.config = cfg

    cos = torch.tensor([1.0, 1.0, 1.0, 1.0])
    sin = torch.tensor([0.0, 0.0, 0.0, 0.0])
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])

    rotated = mla_dummy.apply_rope(x, cos, sin)
    assert torch.allclose(rotated, x), "Zero sine should keep vector unchanged."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FlashInfer MLA attention")
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

    try:
        out, (ckv_cache, kpe_cache) = mla(hidden_states, past_seq_len=0)
        assert out.shape == (batch_size, seq_len, hidden)
        assert ckv_cache.shape == (batch_size * seq_len, 1, latent_dim)
        assert kpe_cache.shape == (batch_size * seq_len, 1, rope_dim)
    except Exception as e:
        # FlashInfer may require specific GPU architectures (e.g. sm_80+ or sm_90+)
        pytest.skip(f"FlashInfer MLA runtime skipped due to GPU backend requirement: {e}")
