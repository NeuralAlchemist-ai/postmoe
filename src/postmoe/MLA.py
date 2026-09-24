import torch
import flashinfer
from flashinfer.prefill import single_prefill_with_kv_cache

class MLA_Attention():
    def __init__(self, kv_down_proj, k_up_proj, v_up_proj, q_nope, q_rope, k_rope, o_proj):
        super().__init__()

        self.kv_down_proj = kv_down_proj
        self.k_up_proj = k_up_proj
        self.v_up_proj = v_up_proj

        self.q_nope = q_nope
        self.q_rope = q_rope

        self.k_rope = k_rope

        self.o_proj = o_proj

    def forward(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
    ):

        batch_size, seq_len, _ = hidden_states.shape

        # -------------------------
        # 1. Current latent KV
        # -------------------------

        current_c_kv = self.kv_down_proj(hidden_states)

        if past_key_values is not None:
            c_kv = torch.cat(
                [past_key_values[0], current_c_kv],
                dim=1
            )
        else:
            c_kv = current_c_kv

        total_seq_len = c_kv.shape[1]

        # -------------------------
        # 2. Reconstruct K_nope/V
        # -------------------------

        v = self.v_up_proj(c_kv)

        v = v.view(
            batch_size,
            total_seq_len,
            self.config.n_head,
            self.config.head_dim
        ).transpose(1, 2)

        # -------------------------
        # 3. Current Q
        # -------------------------

        q_nope = self.q_nope(hidden_states)
        q_rope = self.q_rope(hidden_states)

        q_nope = q_nope.view(
            batch_size,
            seq_len,
            self.config.n_head,
            self.config.nope_dim
        ).transpose(1, 2)

        q_rope = q_rope.view(
            batch_size,
            seq_len,
            self.config.n_head,
            self.config.rope_dim
        ).transpose(1, 2)

        # -------------------------
        # 4. Current K_rope
        # -------------------------

        current_k_rope = self.k_rope(hidden_states)

        current_k_rope = current_k_rope.view(
            batch_size,
            seq_len,
            self.config.n_head,
            self.config.rope_dim
        ).transpose(1, 2)

        if past_key_values is not None:
            k_rope = torch.cat(
                [past_key_values[1], current_k_rope],
                dim=2
            )
        else:
            k_rope = current_k_rope

        # -------------------------
        # 5. Attention
        # -------------------------

        W_UK = self.k_up_proj.weight.view(
            self.config.n_head,
            self.config.nope_dim,
            self.config.kv_latent_dim
        )

        q_projected = torch.einsum(
            "bhld,hdm->bhlm",
            q_nope,
            W_UK
        )

        scores_nope = q_projected @ c_kv.transpose(-2, -1)

        scores_rope = q_rope @ k_rope.transpose(-2, -1)

        scores = scores_nope + scores_rope

        scores_rope = (
            q_rope @ k_rope.transpose(-2, -1)
        )
        
        scores = scores_nope + scores_rope

        scores = scores / (
            (self.nope_dim+self.rope_dim) ** 0.5
        )

        att = F.softmax(scores, dim=-1)

        # -------------------------
        # 6. Apply V
        # -------------------------

        out = att @ v

        # -------------------------
        # 7. Output
        # -------------------------

        out = out.transpose(1, 2)
        out = out.contiguous().view(
            batch_size,
            seq_len,
            -1
        )

        out = self.o_proj(out)

        # -------------------------
        # 8. Return cache
        # -------------------------

        present_c_kv = c_kv
        present_k_rope = k_rope

        return out, present_c_kv, present_k_rope