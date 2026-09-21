import torch
import torch.nn as nn

class UniversalAttentionDecoupler(nn.Module):
    def __init__(self, config, rope_dim_per_head, split_strategy="suffix"):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.num_kv_heads = config.num_key_value_heads
        self.num_q_heads = config.num_attention_heads
        self.gqa_group_size = self.num_q_heads // self.num_kv_heads
        
        self.total_head_dim = self.hidden_size // self.num_q_heads 
        if hasattr(config, "kv_channels"): 
            self.total_head_dim = config.kv_channels
            
        self.rope_dim = rope_dim_per_head
        self.split_strategy = split_strategy
        
        assert self.rope_dim >= 0, f"rope_dim ({self.rope_dim}) can not be greater than total_head_dim ({self.total_head_dim})"

    def extract_weights(self, k_proj_weights):
        """
        Extract the ROPE and NOPE weights from the k_proj_weights based on the specified split strategy.
        """
        device = k_proj_weights.device
        dtype = k_proj_weights.dtype

        W_k = k_proj_weights.view(self.num_kv_heads, self.total_head_dim, self.hidden_size)

        if self.split_strategy == "suffix":
            W_k_rope = W_k[:, -self.rope_dim:, :]
            W_k_nope = W_k[:, :-self.rope_dim, :]

        elif self.split_strategy == "prefix":
            W_k_rope = W_k[:, :self.rope_dim, :]
            W_k_nope = W_k[:, self.rope_dim:, :]

        elif self.split_strategy == "interleaved":
            stride = self.total_head_dim // self.rope_dim
            indices = torch.arange(self.total_head_dim, device=device, dtype=dtype)
            rope_indices = (indices % stride == 0) & (indices < self.rope_dim * stride)
            W_k_rope = W_k[:, rope_indices, :]
            W_k_nope = W_k[:, ~rope_indices, :]
            
        else:
            raise ValueError(f"Invalid split_strategy: {self.split_strategy}. Must be 'prefix' or 'suffix' or 'interleaved'.")

        W_k_rope_flat = W_k_rope.reshape(self.num_kv_heads * W_k_rope.shape[1], self.hidden_size)
        W_k_nope_flat = W_k_nope.reshape(self.num_kv_heads * W_k_nope.shape[1], self.hidden_size)

        return W_k_nope_flat, W_k_rope_flat

    def convert_to_target_layout(self, exctracted_tensors, repeat_mode = 'gqa_brodcast'):
        """
        Convert the extracted ROPE and NOPE tensors to the target layout based on the specified repeat mode.
        """
        W_k_nope, W_k_rope = exctracted_tensors

        if repeat_mode == None:
            return W_k_nope, W_k_rope
        W_k_nope_headed = W_k_nope.view(self.num_kv_heads, -1, self.hidden_size)
        W_k_rope_headed = W_k_rope.view(self.num_kv_heads, -1, self.hidden_size)

        if repeat_mode == "gqa_broadcast":#Converting GQA to MLA
            W_k_nope_target = W_k_nope_headed.repeat_interleave(self.gqa_group_size, dim=0)
            W_k_rope_target = W_k_rope_headed.repeat_interleave(self.gqa_group_size, dim=0)
            
        else:
            raise ValueError(f"Invalid repeat_mode: {repeat_mode}. Must be 'gqa_broadcast' or None.")

        W_k_nope_target = W_k_nope_target.reshape(self.num_q_heads, -1, self.hidden_size)
        W_k_rope_target = W_k_rope_target.reshape(self.num_q_heads, -1, self.hidden_size)
        return W_k_nope_target, W_k_rope_target