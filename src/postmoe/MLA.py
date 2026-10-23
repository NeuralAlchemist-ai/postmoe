import torch
import torch.nn.functional as F

class MLA_Attention(torch.nn.Module):
    def __init__(
        self,
        config,
        kv_down_proj,
        k_up_proj,
        v_up_proj,
        q_nope,
        q_rope,
        k_rope,
        o_proj,
        use_flashinfer: bool = False,
        **kwargs,
    ):
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
        cos = cos.to(dtype=x.dtype, device=x.device)
        sin = sin.to(dtype=x.dtype, device=x.device)
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]

        x_rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (x_rotated * sin)

    def _infer_past_seq_len(self, past_seq_len, past_key_values, past_key_value):
        """HF never passes past_seq_len; derive it from cache when present."""
        if past_seq_len is not None:
            return past_seq_len

        cache = past_key_values if past_key_values is not None else past_key_value
        if cache is None:
            return 0

        # Our own cache tuple from a previous forward: (ckv_cache, kpe_cache)
        if isinstance(cache, tuple) and len(cache) == 2 and torch.is_tensor(cache[0]):
            return cache[0].shape[0]

        # HuggingFace DynamicCache / similar
        if hasattr(cache, "get_seq_length"):
            try:
                return int(cache.get_seq_length())
            except TypeError:
                return int(cache.get_seq_length(0))

        return 0

    def _project(self, hidden_states, past_seq_len):
        B, L, _ = hidden_states.shape
        T = B * L
        c = self.config

        cos, sin = self.freq_rope(L, hidden_states.device, past_seq_len)
        cos = cos.repeat(B, 1)
        sin = sin.repeat(B, 1)

        q_nope = self.q_nope(hidden_states).view(T, c.n_head, c.nope_dim)
        q_rope = self.q_rope(hidden_states).view(T, c.n_head, c.rope_dim)
        c_kv = self.kv_down_proj(hidden_states).view(T, c.kv_latent_dim)
        k_rope = self.k_rope(hidden_states).view(T, c.rope_dim)

        q_rope = self.apply_rope(q_rope, cos.unsqueeze(1), sin.unsqueeze(1))
        k_rope = self.apply_rope(k_rope, cos, sin)

        w_uk = self.k_up_proj.weight.view(c.n_head, c.nope_dim, c.kv_latent_dim)
        w_uv = self.v_up_proj.weight.view(c.n_head, c.head_dim, c.kv_latent_dim)
        q_abs = torch.einsum("thd,hdc->thc", q_nope, w_uk).contiguous()

        return B, L, T, q_abs, q_rope, c_kv, k_rope, w_uv

    def _torch_attn(self, B, L, T, q_abs, q_rope, c_kv, k_rope, sm_scale):
        """Absorbed MLA attention (causal) using PyTorch."""
        c = self.config
        # Per-batch sequences stacked as T = B*L contiguous tokens.
        q_abs_b = q_abs.view(B, L, c.n_head, c.kv_latent_dim)
        q_rope_b = q_rope.view(B, L, c.n_head, c.rope_dim)
        c_kv_b = c_kv.view(B, L, c.kv_latent_dim)
        k_rope_b = k_rope.view(B, L, c.rope_dim)

        # scores[b,h,i,j] = <q_abs, c_kv> + <q_rope, k_rope>
        scores = torch.einsum("bihd,bjd->bhij", q_abs_b, c_kv_b)
        scores = scores + torch.einsum("bihd,bjd->bhij", q_rope_b, k_rope_b)
        scores = scores * sm_scale

        causal = torch.triu(
            torch.ones(L, L, device=scores.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_abs.dtype)

        o_lat = torch.einsum("bhij,bjd->bihd", attn, c_kv_b).reshape(
            T, c.n_head, c.kv_latent_dim
        )
        ckv_cache = c_kv.unsqueeze(1)
        kpe_cache = k_rope.unsqueeze(1)
        return o_lat, ckv_cache, kpe_cache

    def forward(
        self,
        hidden_states,
        past_seq_len=None,
        position_embeddings=None,
        past_key_values=None,
        past_key_value=None,
        **kw,
    ):
        # Transformers calls: self_attn(hidden_states, attention_mask=..., past_key_value=..., ...)
        # so past_seq_len must be optional and inferred.
        past_seq_len = self._infer_past_seq_len(
            past_seq_len, past_key_values, past_key_value
        )

        B, L, _ = hidden_states.shape
        if L == 0:
            # WikiText has blank rows; keep shapes valid for the rest of the stack.
            empty = hidden_states.new_zeros(B, 0, hidden_states.shape[-1])
            return empty, None

        c = self.config
        sm_scale = 1.0 / (c.nope_dim + c.rope_dim) ** 0.5
        B, L, T, q_abs, q_rope, c_kv, k_rope, w_uv = self._project(
            hidden_states, past_seq_len
        )

        o_lat, ckv_cache, kpe_cache = self._torch_attn(
            B, L, T, q_abs, q_rope, c_kv, k_rope, sm_scale
        )

        out = torch.einsum("thc,hvc->thv", o_lat, w_uv).reshape(B, L, -1)
        return self.o_proj(out), (ckv_cache, kpe_cache)
