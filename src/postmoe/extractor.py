import torch
import torch.nn as nn

class UniversalAttentionDecoupler(nn.Module):
    def __init__(self, model, config, rope_dim_per_head, split_strategy="suffix"):
        super().__init__()
        self.model = model
        self.hidden_size = config.hidden_size

        self.num_kv_heads = config.num_key_value_heads
        self.num_q_heads = config.num_attention_heads
        self.gqa_group_size = self.num_q_heads // self.num_kv_heads
        self.num_of_layers = config.num_hidden_layers
        
        self.total_head_dim = self.hidden_size // self.num_q_heads 
        if hasattr(config, "kv_channels"): 
            self.total_head_dim = config.kv_channels
            
        self.rope_dim = rope_dim_per_head
        self.split_strategy = split_strategy
        
        assert self.rope_dim >= 0, f"rope_dim ({self.rope_dim}) can not be greater than total_head_dim ({self.total_head_dim})"


    def _universal_qkv_proj_extractor(self):
        """Universal and efficient extractor for k_proj weights (without for loops).

        Returns a stacked 3D tensor of all layer weights. Shape: [num_layers,
        out_features, in_features]
        """
        k_weights = []
        q_weights = []
        v_weights = []

        for layer_idx in range(self.num_of_layers):
            attention = self.model.model.layers[layer_idx].self_attn
            k_weights.append(attention.k_proj.weight.detach())
            q_weights.append(attention.q_proj.weight.detach())
            v_weights.append(attention.v_proj.weight.detach())

        return (
            torch.stack(k_weights),
            torch.stack(v_weights),
            torch.stack(q_weights),
        )


    def _extract_weights(self):
        """
        Extract the ROPE and NOPE weights from the k_proj_weights based on the specified split strategy.
        """
        k_proj_weights, v_proj_weights, q_proj_weights = self._universal_qkv_proj_extractor()
        device = k_proj_weights.device

        W_k = k_proj_weights.view(self.num_of_layers,self.num_kv_heads, self.total_head_dim, self.hidden_size)
        W_q = q_proj_weights.view(self.num_of_layers,self.num_q_heads, self.total_head_dim, self.hidden_size)
        W_v = v_proj_weights.view(self.num_of_layers,self.num_kv_heads, self.total_head_dim, self.hidden_size)
        if self.split_strategy == "suffix":
            W_k_rope = W_k[:, :, -self.rope_dim:, :]
            W_k_nope = W_k[:, :, :-self.rope_dim, :]

            W_q_rope = W_q[:, :, -self.rope_dim:, :]
            W_q_nope = W_q[:, :, :-self.rope_dim, :]

        elif self.split_strategy == "prefix":
            W_k_rope = W_k[:, :, :self.rope_dim, :]
            W_k_nope = W_k[:, :, self.rope_dim:, :]
            W_q_rope = W_q[:, :, :self.rope_dim, :]
            W_q_nope = W_q[:, :, self.rope_dim:, :]

        elif self.split_strategy == "interleaved":
            stride = self.total_head_dim // self.rope_dim
            indices = torch.arange(self.total_head_dim, device=device)
            rope_indices = (indices % stride == 0) & (indices < self.rope_dim * stride)
            W_k_rope = W_k[:, :, rope_indices, :]
            W_k_nope = W_k[:, :, ~rope_indices, :]
            W_q_rope = W_q[:, :, rope_indices, :]
            W_q_nope = W_q[:, :, ~rope_indices, :]
        else:
            raise ValueError(f"Invalid split_strategy: {self.split_strategy}. Must be 'prefix' or 'suffix' or 'interleaved'.")

        return W_k_nope, W_k_rope, W_q_nope, W_q_rope, W_v

    def convert_to_target_layout(self, repeat_mode = 'gqa_broadcast'):
        """
        Convert the extracted ROPE and NOPE tensors to the target layout based on the specified repeat mode.
        """
        W_k_nope, W_k_rope, W_q_nope, W_q_rope, W_v = self._extract_weights()

        if repeat_mode == None:
            return W_k_nope, W_k_rope, W_q_nope, W_q_rope, W_v
        W_k_nope_headed = W_k_nope.view(
            self.num_of_layers,
            self.num_kv_heads,
            -1,
            self.hidden_size,
        )
        W_k_rope_headed = W_k_rope.view(
            self.num_of_layers,
            self.num_kv_heads,
            -1,
            self.hidden_size,
        )

        if repeat_mode == "gqa_broadcast":#Converting GQA to MLA
            W_k_nope_target = W_k_nope_headed.repeat_interleave(
                self.gqa_group_size,
                dim=1,
            )
            W_k_rope_target = W_k_rope_headed.repeat_interleave(
                self.gqa_group_size,
                dim=1,
            )
            
        else:
            raise ValueError(f"Invalid repeat_mode: {repeat_mode}. Must be 'gqa_broadcast' or None.")

        return W_k_nope_target, W_k_rope_target, W_q_nope, W_q_rope, W_v