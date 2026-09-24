from flashinfer.mla._batch_mla._backends import cutile_backend
import torch
import flashinfer
# Для MLA во FlashInfer используется специализированный Batch API Wrapper
from flashinfer.mla import BatchMLAPagedAttentionWrapper

class MLA_Attention(torch.nn.Module):
    def __init__(self, config, kv_down_proj, k_up_proj, v_up_proj, q_nope, q_rope, k_rope, o_proj):
        super().__init__()
        self.config = config
        
        self.kv_down_proj = kv_down_proj
        self.k_up_proj = k_up_proj
        self.v_up_proj = v_up_proj

        self.q_nope = q_nope
        self.q_rope = q_rope
        self.k_rope = k_rope
        self.o_proj = o_proj

        self.bmm2_scale = 1.0
        
        # Инициализируем обертку FlashInfer MLA
        # Движку требуется рабочая область (workspace) на GPU для промежуточных вычислений
        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda") # 128MB
        self.flashinfer_mla = BatchMLAPagedAttentionWrapper(self.workspace_buffer)

        cos_cache, sin_cache = self._build_rope_cache(
            config.max_seq_len, self.config.rope_dim
        )
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def _build_rope_cache(self, max_seq_len: int, rope_dim: int, theta: int = 10000):
        inv_freq = 1 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        # Duplicated (not interleaved) so the layout matches the half-split
        # rotation in MultiHeadAttention.apply_rope.
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def freq_rope(self, seq_len: int, device: torch.device, past_seq_len: int = 0):
        total_len = past_seq_len + seq_len
        if total_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {total_len} exceeds max_seq_len "
                f"({self.config.max_seq_len})"
            )
        cos = self.cos_cache[past_seq_len:total_len].to(device)
        sin = self.sin_cache[past_seq_len:total_len].to(device)
        return cos, sin

    def apply_rope(self, x, cos, sin):
            # cos/sin are (T, head_size) and broadcast over (B, n_heads, T, head_size)
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]

            x_rotated = torch.cat((-x2, x1), dim=-1)
            return (x * cos) + (x_rotated * sin)

    def forward(self, hidden_states, past_seq_len, position_embeddings=None, past_key_values=None, **kw):
        B, L, _ = hidden_states.shape
        T = B * L
        c = self.config
        sm_scale = 1.0 / (c.nope_dim + c.rope_dim) ** 0.5

        cos, sin = self.freq_rope(L, hidden_states.device, past_seq_len)
        cos = cos.repeat(B,1)
        sin = sin.repeat(B,1)

        q_nope = self.q_nope(hidden_states).view(T, c.n_head, c.nope_dim)
        q_rope = self.q_rope(hidden_states).view(T, c.n_head, c.rope_dim)
        c_kv   = self.kv_down_proj(hidden_states).view(T, c.kv_latent_dim)   # + norm
        k_rope = self.k_rope(hidden_states).view(T, c.rope_dim)              # k_rope: Linear(hidden, rope_dim)

        q_rope = self.apply_rope(q_rope, cos.unsqueeze(1), sin.unsqueeze(1))
        k_rope = self.apply_rope(k_rope, cos, sin)

        # абсорбция W_UK в запрос
        w_uk = self.k_up_proj.weight.view(c.n_head, c.nope_dim, c.kv_latent_dim)
        w_uv = self.v_up_proj.weight.view(c.n_head, c.head_dim, c.kv_latent_dim)
        q_abs = torch.einsum("thd,hdc->thc", q_nope, w_uk).contiguous()      # [T, H, latent]

        # кэш: page_size=1 для простоты; в реальности храните пулы страниц
        ckv_cache = c_kv.unsqueeze(1)      # [T, 1, latent]
        kpe_cache = k_rope.unsqueeze(1)    # [T, 1, rope]

        qo_indptr = torch.arange(0, B + 1, device="cuda", dtype=torch.int32) * L
        kv_indptr = qo_indptr.clone()
        kv_indices = torch.arange(T, device="cuda", dtype=torch.int32)
        kv_len_arr = torch.full((B,), L, dtype=torch.int32, device="cuda")

        self.flashinfer_mla.plan(
            qo_indptr, kv_indptr, kv_indices, kv_len_arr,
            c.n_head, c.kv_latent_dim, c.rope_dim, 1,
            True, sm_scale, q_abs.dtype, ckv_cache.dtype,
        )
        o_lat = self.flashinfer_mla.run(q_abs, q_rope, ckv_cache, kpe_cache)  # [T, H, latent]

        # абсорбция W_UV на выходе
        out = torch.einsum("thc,hvc->thv", o_lat, w_uv).reshape(B, L, -1)
        return self.o_proj(out), (ckv_cache, kpe_cache)