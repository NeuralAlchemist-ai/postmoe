import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from postmoe.converter import DataDrivenSVD

class MLA_Attention(nn.Module):
    def __init__(
        self,
        config,
        kv_down_proj: nn.Linear,
        k_up_proj: nn.Linear,
        v_up_proj: nn.Linear,
        q_nope: nn.Linear,
        q_rope: nn.Linear,
        k_rope: nn.Linear,
        o_proj: nn.Linear,
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

        self.n_head = config.n_head
        self.nope_dim = config.nope_dim
        self.rope_dim = config.rope_dim
        self.kv_latent_dim = config.kv_latent_dim
        self.head_dim = config.head_dim

    def _get_rope_embeddings(self, position_embeddings, seq_len: int, device: torch.device, past_seq_len: int = 0):
        total_len = past_seq_len + seq_len
        if position_embeddings is not None:
            if hasattr(position_embeddings, "cos_cached"):
                cos = position_embeddings.cos_cached[:total_len].to(device)
                sin = position_embeddings.sin_cached[:total_len].to(device)
            else:
                cos, sin = position_embeddings(total_len, device=device)
            cos = cos[past_seq_len:total_len]
            sin = sin[past_seq_len:total_len]
        else:
            theta = getattr(self.config, "rope_theta", 10000.0)
            inv_freq = 1.0 / (theta ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32, device=device) / self.rope_dim))
            t = torch.arange(total_len, dtype=torch.float32, device=device)[past_seq_len:total_len]
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
            
        return cos, sin

    def apply_rope(self, x, cos, sin):
        if cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif cos.ndim == 1:
            while cos.ndim < x.ndim:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
        cos = cos.to(dtype=x.dtype, device=x.device)
        sin = sin.to(dtype=x.dtype, device=x.device)
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (x_rotated * sin)

    def _infer_past_seq_len(self, past_key_values, **kwargs):
        if "past_seq_len" in kwargs and kwargs["past_seq_len"] is not None:
            return kwargs["past_seq_len"]
        cache = past_key_values if past_key_values is not None else kwargs.get("past_key_value", None)
        if cache is None:
            return 0
        if isinstance(cache, tuple) and len(cache) == 2 and cache[0] is not None and torch.is_tensor(cache[0]):
            return cache[0].shape[1]  # S dimension from [B, S, kv_latent_dim]
        if hasattr(cache, "get_seq_length"):
            try:
                return cache.get_seq_length()
            except TypeError:
                return cache.get_seq_length(0)
        return 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[nn.Module] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        B, L, H_dim = hidden_states.shape
        if L == 0:
            return hidden_states.new_zeros(B, 0, H_dim), past_key_values

        past_seq_len = self._infer_past_seq_len(past_key_values, **kwargs)
        total_kv_len = past_seq_len + L

        # 1. Извлекаем проекции Query и вращаем RoPE часть
        q_nope = self.q_nope(hidden_states).view(B, L, self.n_head, self.nope_dim)
        q_rope = self.q_rope(hidden_states).view(B, L, self.n_head, self.rope_dim)

        cos, sin = self._get_rope_embeddings(position_embeddings, L, hidden_states.device, past_seq_len)
        q_rope = self.apply_rope(q_rope, cos, sin)

        # 2. Извлекаем базовые KV проекции
        c_kv = self.kv_down_proj(hidden_states) # [B, L, kv_latent_dim]
        k_rope = self.k_rope(hidden_states).view(B, L, 1, self.rope_dim) 
        k_rope = self.apply_rope(k_rope, cos, sin)

        # 3. Динамический KV-кэш
        if past_key_values is not None:
            old_c_kv, old_k_rope = past_key_values
            c_kv = torch.cat([old_c_kv, c_kv], dim=1)
            k_rope = torch.cat([old_k_rope, k_rope], dim=1)
        
        new_kv_cache = (c_kv, k_rope) if use_cache else None
        S = c_kv.shape[1] # Полная длина ключей/значений

        # 4. ПОДГОТОВКА К ТРЮКУ С КОНКАТЕНАЦИЕЙ ДЛЯ FlashAttention / SDPA
        w_uk = self.k_up_proj.weight.view(self.n_head, self.nope_dim, self.kv_latent_dim)
        
        # Поглощаем W_UK в Query: [B, L, n_head, kv_latent_dim]
        q_abs = torch.einsum("blhd,hdc->blhc", q_nope, w_uk)
        
        # Объединяем nope и rope пространства по последней размерности (D)
        # На выходе получаем классический формат для SDPA: [B, n_head, L, D_total]
        query_states = torch.cat([q_abs, q_rope], dim=-1).transpose(1, 2)
        
        # Конкатенируем ключи. Так как c_kv общий для всех голов, размножаем его по оси H.
        k_nope_expanded = c_kv.unsqueeze(2).expand(-1, -1, self.n_head, -1)
        k_rope_expanded = k_rope.expand(-1, -1, self.n_head, -1)
        key_states = torch.cat([k_nope_expanded, k_rope_expanded], dim=-1).transpose(1, 2) # [B, n_head, S, D_total]

        # 5. МАСКА И МАСШТАБИРОВАНИЕ (Scale)
        # Во FlashAttention коэффициент масштабирования применяется автоматически как 1 / sqrt(D).
        # Но у нас D_total = kv_latent_dim + rope_dim. По спецификации MLA коэффициент равен:
        # 1 / sqrt(nope_dim + rope_dim). Передаем кастомный scale_factor в SDPA.
        custom_scale = 1.0 / ((self.nope_dim + self.rope_dim) ** 0.5)

        # Логика причинно-следственной (causal) маски
        is_causal = True if L > 1 and past_seq_len == 0 else False
        
        # Настройка маски паддинга (attention_mask из HF идет со скорами или True/False)
        # Обратите внимание: если у вас 2D маска [B, S], SDPA может потребовать формат bool инвертированный или float
        if attention_mask is not None and attention_mask.dtype in (torch.float32, torch.float16, torch.bfloat16):
            # Если маска аддитивная (содержит значения -inf), Flash/SDPA примет её только в кастомном режиме
            is_causal = False
        
        # 6. ВЫЗОВ НАТИВНОГО СКОРОСТНОГО БЛОКА ВНИМАНИЯ (SDPA / FlashAttention)
        # Так как реальных весов V у нас нет (они заменяются матрицей W_UV на выходе),
        # в качестве фиктивного "Value" для SDPA мы отдаем сам латентный кэш c_kv,
        # размноженный по головам! На выходе мы получим o_lat без материализации матрицы скоров в VRAM.
        value_states = c_kv.unsqueeze(2).expand(-1, -1, self.n_head, -1).transpose(1, 2) # [B, n_head, S, kv_latent_dim]

        o_lat = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=custom_scale
        ) # Выход имеет размерность: [B, n_head, L, kv_latent_dim]

        # Переводим обратно в формат токенов: [B, L, n_head, kv_latent_dim]
        o_lat = o_lat.transpose(1, 2)

        # 7. ФИНАЛЬНАЯ ПРОЕКЦИЯ С ВЫХОДНЫМИ ВЕСАМИ ЗНАЧЕНИЙ (Value W_UV)
        w_uv = self.v_up_proj.weight.view(self.n_head, self.head_dim, self.kv_latent_dim)
        
        # Проецируем сжатый латентный контекст обратно в полный размер голов
        context_states = torch.einsum("blhc,hvc->blhv", o_lat, w_uv).reshape(B, L, -1)

        # o_proj (Линейный слой, преобразующий обратно в hidden_size модели)
        attn_output = self.o_proj(context_states)

        return attn_output, new_kv_cache



class VLLMArchitectureConverter:
    def __init__(self, config):
        """
        Args:
            config: Configuration of original model (e.g. LlamaConfig, Qwen2Config, or postmoe Config)
        """
        self.config = config
        self.n_head = getattr(config, "num_attention_heads", getattr(config, "n_head", 16))
        self.head_dim = getattr(config, "head_dim", 64)
        self.hidden_size = getattr(config, "hidden_size", self.n_head * self.head_dim)
        
        self.kv_latent_dim = getattr(config, "kv_latent_dim", 512)
        self.nope_dim = getattr(config, "nope_dim", 128)
        self.rope_dim = getattr(config, "rope_dim", 64)
        if hasattr(config, "nope_dim") and hasattr(config, "rope_dim"):
            self.head_dim = getattr(config, "head_dim", self.nope_dim + self.rope_dim)

    def convert_attention_layer_to_vllm(
        self, 
        w_q: torch.Tensor, 
        w_k: torch.Tensor, 
        w_v: torch.Tensor, 
        w_o: torch.Tensor,
        cov_matrix: Optional[torch.Tensor] = None,
        target_rank: Optional[int] = None,
        split_strategy: str = "suffix",
    ) -> Dict[str, torch.Tensor]:
        """
        Converts one raw MHA/GQA layer to MLA weights formatted for vLLM.
        Decouples Q and K into NOPE and RoPE components (using the decoupler strategy),
        averages K RoPE across heads into a single shared projection,
        and applies DataDrivenSVD compression to K_nope and V.
        """
        n_kv_head = getattr(self.config, "num_key_value_heads", self.n_head)
        total_head_dim = self.hidden_size // self.n_head
        if hasattr(self.config, "kv_channels"):
            total_head_dim = self.config.kv_channels

        w_q_headed = w_q.view(self.n_head, total_head_dim, self.hidden_size)
        w_k_headed = w_k.view(n_kv_head, total_head_dim, self.hidden_size)
        w_v_headed = w_v.view(n_kv_head, total_head_dim, self.hidden_size)

        if split_strategy == "suffix":
            w_q_rope = w_q_headed[:, -self.rope_dim:, :]
            w_q_nope = w_q_headed[:, :-self.rope_dim, :]
            w_k_rope = w_k_headed[:, -self.rope_dim:, :]
            w_k_nope = w_k_headed[:, :-self.rope_dim, :]
        elif split_strategy == "prefix":
            w_q_rope = w_q_headed[:, :self.rope_dim, :]
            w_q_nope = w_q_headed[:, self.rope_dim:, :]
            w_k_rope = w_k_headed[:, :self.rope_dim, :]
            w_k_nope = w_k_headed[:, self.rope_dim:, :]
        elif split_strategy == "interleaved":
            stride = total_head_dim // self.rope_dim
            indices = torch.arange(total_head_dim, device=w_q.device)
            rope_indices = (indices % stride == 0) & (indices < self.rope_dim * stride)
            w_q_rope = w_q_headed[:, rope_indices, :]
            w_q_nope = w_q_headed[:, ~rope_indices, :]
            w_k_rope = w_k_headed[:, rope_indices, :]
            w_k_nope = w_k_headed[:, ~rope_indices, :]
        else:
            raise ValueError(f"Invalid split_strategy: {split_strategy}")

        # Broadcast GQA to Q heads
        gqa_group_size = self.n_head // n_kv_head
        w_k_nope_broadcast = w_k_nope.repeat_interleave(gqa_group_size, dim=0).reshape(-1, self.hidden_size)
        w_v_broadcast = w_v_headed.repeat_interleave(gqa_group_size, dim=0).reshape(-1, self.hidden_size)

        # Single shared RoPE key projection across all heads
        w_k_rope_shared = w_k_rope.mean(dim=0)  # [rope_dim, hidden_size]

        w_q_nope_2d = w_q_nope.reshape(-1, self.hidden_size)
        w_q_rope_2d = w_q_rope.reshape(-1, self.hidden_size)

        return self.convert_decoupled_layer_to_vllm(
            w_q_nope=w_q_nope_2d,
            w_q_rope=w_q_rope_2d,
            w_k_nope=w_k_nope_broadcast,
            w_k_rope=w_k_rope_shared,
            w_v=w_v_broadcast,
            w_o=w_o,
            cov_matrix=cov_matrix,
            target_rank=target_rank,
        )

    def convert_decoupled_layer_to_vllm(
        self,
        w_q_nope: torch.Tensor,
        w_q_rope: torch.Tensor,
        w_k_nope: torch.Tensor,
        w_k_rope: torch.Tensor,
        w_v: torch.Tensor,
        w_o: torch.Tensor,
        cov_matrix: Optional[torch.Tensor] = None,
        target_rank: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Converts already-decoupled components (e.g. from UniversalAttentionDecoupler) to vLLM format.
        Uses DataDrivenSVD to compress K_nope and V into (kv_down, k_up, v_up).
        """
        rank = target_rank if target_rank is not None else self.kv_latent_dim
        svd_converter = DataDrivenSVD(w_k_nope, w_v)
        w_kv_down, w_k_up, w_v_up = svd_converter.convert(
            target_rank=rank,
            cov_matrix=cov_matrix,
        )

        expected_o_in = self.n_head * self.head_dim
        if w_o.shape[1] == expected_o_in:
            w_o_new = w_o
        else:
            w_o_new = torch.zeros(
                self.hidden_size, expected_o_in, dtype=w_o.dtype, device=w_o.device
            )
            orig_head_dim = w_o.shape[1] // self.n_head
            safe_head_dim = min(orig_head_dim, self.head_dim)
            for h in range(self.n_head):
                w_o_new[:, h * self.head_dim : h * self.head_dim + safe_head_dim] = (
                    w_o[:, h * orig_head_dim : h * orig_head_dim + safe_head_dim]
                )

        vllm_layer_weights = {
            "q_proj.weight": w_q_nope,
            "q_rope_proj.weight": w_q_rope,
            "kv_down_proj.weight": w_kv_down,  # [kv_latent_dim, hidden_size]
            "k_up_proj.weight": w_k_up,        # [n_head * nope_dim, kv_latent_dim]
            "v_up_proj.weight": w_v_up,        # [n_head * head_dim, kv_latent_dim]
            "k_rope_proj.weight": w_k_rope,    # [rope_dim, hidden_size]
            "o_proj.weight": w_o_new,
        }
        return vllm_layer_weights

    def generate_vllm_config(self, save_path: str, num_experts: int = 0, top_k: int = 0):
        """
        Creates config.json for running the converted DeepSeekV2 architecture in vLLM.
        """
        vllm_config = {
            "architectures": ["DeepseekV2ForCausalLM"],
            "model_type": "deepseek_v2",
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.n_head,
            "num_hidden_layers": getattr(self.config, "num_hidden_layers", 1),
            "vocab_size": getattr(self.config, "vocab_size", 32000),
            "kv_latent_dim": self.kv_latent_dim,
            "nope_dim": self.nope_dim,
            "rope_dim": self.rope_dim,
            "q_lora_rank": 0,
            "n_routed_experts": num_experts,
            "num_experts_per_tok": top_k,
            "moe_layer_freq": 1,
            "max_position_embeddings": getattr(self.config, "max_position_embeddings", 4096),
            "rope_theta": getattr(self.config, "rope_theta", 10000.0),
            "torch_dtype": str(getattr(self.config, "torch_dtype", "bfloat16")).split(".")[-1],
        }

        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(vllm_config, f, indent=2)


def run_converter_pipeline(source_model_name_or_dir: str, output_dir: str, num_experts: int = 0, top_k: int = 0):
    """
    End-to-end pipeline to convert a model to vLLM DeepSeek format using UniversalAttentionDecoupler.
    """
    from pipeline import Pipeline
    from postmoe.config import Config

    cfg = Config(model_name=source_model_name_or_dir)
    pipeline = Pipeline(cfg)
    pipeline.convert()
    pipeline.export_to_vllm(output_dir, num_experts=num_experts, top_k=top_k)
