import os
import torch
import torch.nn as nn
from dataclasses import replace

from postmoe.eval import Evaluation
from postmoe.extractor import UniversalAttentionDecoupler
from postmoe.converter import Converter, DataDrivenSVD, LayerActivationAccumulator
from postmoe.MLA import MLA_Attention, VLLMArchitectureConverter

from loader.load_model import ModelLoader
from loader.data_loader import DataLoader


def _make_linear(weight_2d, dtype=None, device=None):
    """Wrap a 2D weight tensor [out_features, in_features] into nn.Linear(bias=False)."""
    out_f, in_f = weight_2d.shape
    weight = weight_2d.detach()
    if dtype is not None:
        weight = weight.to(dtype=dtype)
    if device is not None:
        weight = weight.to(device=device)
    lin = nn.Linear(in_f, out_f, bias=False, device=weight.device, dtype=weight.dtype)
    lin.weight = nn.Parameter(weight)
    return lin


class Pipeline:
    def __init__(self, config):
        self.UniversalAttentionDecoupler = UniversalAttentionDecoupler
        self.Converter = DataDrivenSVD
        self.DataDrivenSVD = DataDrivenSVD
        self.LayerActivationAccumulator = LayerActivationAccumulator
        self.Evaluation = Evaluation
        self.MLA_Attention = MLA_Attention
        self.VLLMArchitectureConverter = VLLMArchitectureConverter

        self.ModelLoader = ModelLoader
        self.DataLoader = DataLoader

        self.model_name = config.model_name
        self.rope_dim = config.rope_dim
        self.mla_config = config           # the postmoe Config dataclass
        self.dataset_name = config.dataset_name
        self.dataset_config = config.dataset_config
        self.split = config.split

        self.model = None
        self.tokenizer = None
        self.hf_config = None              # HuggingFace model config
        self.data = None
        self.all_weights = None            # extracted per-layer weights (tuple of stacked tensors)
        self.activation_accumulator = None

    def convert(self, collect_activations: bool = False, num_calib_samples: int = 16, calib_seq_len: int = 128):
        model_loader = self.ModelLoader(self.model_name)
        model_loader.load_model()
        self.model = model_loader.get_model()
        self.tokenizer = model_loader.get_tokenizer()
        self.hf_config = model_loader.get_config()

        data_loader = self.DataLoader(
            self.dataset_name,
            self.dataset_config,
            self.split,
        )
        self.data = data_loader.load_data()

        if collect_activations:
            self.collect_activations(num_samples=num_calib_samples, max_seq_len=calib_seq_len)

        # Extract and decouple weights for ALL layers at once
        decoupler = self.UniversalAttentionDecoupler(
            self.model, self.hf_config, self.rope_dim
        )
        self.all_weights = decoupler.convert_to_target_layout(
            repeat_mode="gqa_broadcast"
        )
        self._sync_mla_config_from_decoupler(decoupler)

    def collect_activations(self, num_samples: int = 16, max_seq_len: int = 128):
        """Collect activation covariances across attention layers using calibration samples."""
        if self.model is None or self.tokenizer is None or self.data is None:
            raise ValueError("Model, tokenizer, and data must be loaded before collecting activations.")

        accumulator = self.LayerActivationAccumulator(self.model)
        device = next(self.model.parameters()).device
        count = 0
        for row in self.data:
            text = row.get("text") or row.get("content") or ""
            if not isinstance(text, str) or len(text.strip()) < 10:
                continue
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            ).to(device)
            with torch.no_grad():
                self.model(**inputs)
            count += 1
            if count >= num_samples:
                break
        accumulator.remove_hooks()
        self.activation_accumulator = accumulator

    def _sync_mla_config_from_decoupler(self, decoupler):
        """Align MLA dims with the loaded HF model (defaults in Config are placeholders)."""
        total_head_dim = decoupler.total_head_dim
        nope_dim = total_head_dim - self.rope_dim
        self.mla_config = replace(
            self.mla_config,
            n_head=decoupler.num_q_heads,
            nope_dim=nope_dim,
            rope_dim=self.rope_dim,
            head_dim=total_head_dim,
        )

    def custom_MLA_attention(self, layer_idx, o_proj):
        """Build an MLA_Attention module for a single layer.

        Args:
            layer_idx: Index into the stacked weight tensors (layer number).
            o_proj: The original output projection (nn.Linear) from this layer.
        """
        W_k_nope_all, W_k_rope_all, W_q_nope_all, W_q_rope_all, W_v_all = self.all_weights

        # ── Index into this layer ──
        # Shapes after indexing:
        #   k_nope_i:  [num_q_heads, nope_dim, hidden]  (already broadcast from KV→Q heads)
        #   k_rope_i:  [rope_dim, hidden]               (shared RoPE key)
        #   q_nope_i:  [num_q_heads, nope_dim, hidden]
        #   q_rope_i:  [num_q_heads, rope_dim, hidden]
        #   v_i:       [num_q_heads, head_dim, hidden]
        k_nope_i = W_k_nope_all[layer_idx]
        k_rope_i = W_k_rope_all[layer_idx]
        q_nope_i = W_q_nope_all[layer_idx]
        q_rope_i = W_q_rope_all[layer_idx]
        v_i      = W_v_all[layer_idx]

        num_q_heads = q_nope_i.shape[0]
        nope_dim = q_nope_i.shape[1]
        rope_dim = q_rope_i.shape[1]
        head_dim = v_i.shape[1]
        hidden = q_nope_i.shape[-1]

        # ── Expand V to match num_q_heads (if not already broadcast) ──
        if v_i.shape[0] != num_q_heads:
            gqa_group = num_q_heads // v_i.shape[0]
            v_expanded = v_i.repeat_interleave(gqa_group, dim=0)   # [num_q_heads, head_dim, hidden]
        else:
            v_expanded = v_i

        # ── Flatten heads → 2D for SVD ──
        k_nope_2d = k_nope_i.reshape(-1, hidden)   # [num_q_heads * nope_dim, hidden]
        v_2d      = v_expanded.reshape(-1, hidden)  # [num_q_heads * head_dim, hidden]

        # ── Retrieve covariance matrix if collected ──
        cov = None
        if self.activation_accumulator is not None:
            layer_name = f"model.layers.{layer_idx}.self_attn"
            if layer_name in self.activation_accumulator.cov_matrices:
                try:
                    cov = self.activation_accumulator.get_covariance(layer_name)
                except ValueError:
                    cov = None

        target_rank = getattr(self.mla_config, "kv_latent_dim", None)

        # ── Joint SVD compression of K_nope and V with DataDrivenSVD ──
        if hasattr(self.Converter, "convert") or self.Converter is DataDrivenSVD:
            svd_converter = self.DataDrivenSVD(k_nope_2d, v_2d)
            kv_down, k_up, v_up = svd_converter.convert(target_rank=target_rank, cov_matrix=cov)
            rank = kv_down.shape[0]
        else:
            # Fallback to single-matrix Converter
            k_up, kv_down = self.Converter(k_nope_2d).svd_compress(target_rank=target_rank)
            rank = kv_down.shape[0]
            v_up, _ = self.Converter(v_2d).svd_compress(target_rank=rank)

        # ── For MLA, k_rope is a single shared projection across all heads ──
        if k_rope_i.ndim == 3:
            k_rope_single = k_rope_i.mean(dim=0)   # [rope_dim, hidden]
        else:
            k_rope_single = k_rope_i                # already [rope_dim, hidden]

        # ── Flatten Q heads → 2D ──
        q_nope_2d = q_nope_i.reshape(-1, hidden)   # [num_q_heads * nope_dim, hidden]
        q_rope_2d = q_rope_i.reshape(-1, hidden)   # [num_q_heads * rope_dim, hidden]

        # Per-layer config: dims from extracted weights, rank from SVD
        layer_config = replace(
            self.mla_config,
            n_head=num_q_heads,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            head_dim=head_dim,
            kv_latent_dim=rank,
        )

        dtype = o_proj.weight.dtype
        device = o_proj.weight.device

        # ── Wrap in nn.Linear and build MLA_Attention ──
        mla = self.MLA_Attention(
            config=layer_config,
            kv_down_proj=_make_linear(kv_down, dtype=dtype, device=device),
            k_up_proj=_make_linear(k_up, dtype=dtype, device=device),
            v_up_proj=_make_linear(v_up, dtype=dtype, device=device),
            q_nope=_make_linear(q_nope_2d, dtype=dtype, device=device),
            q_rope=_make_linear(q_rope_2d, dtype=dtype, device=device),
            k_rope=_make_linear(k_rope_single, dtype=dtype, device=device),
            o_proj=o_proj,
        )
        return mla.to(device=device, dtype=dtype)

    def export_to_vllm(self, output_dir: str, num_experts: int = 0, top_k: int = 0):
        """Export converted model weights and config.json in vLLM / DeepSeek format."""
        os.makedirs(output_dir, exist_ok=True)
        config_to_use = self.hf_config if self.hf_config else self.mla_config
        vllm_converter = self.VLLMArchitectureConverter(config_to_use)
        vllm_state_dict = {}

        W_k_nope_all, W_k_rope_all, W_q_nope_all, W_q_rope_all, W_v_all = self.all_weights
        num_layers = W_k_nope_all.shape[0]

        for layer_idx in range(num_layers):
            cov = None
            if self.activation_accumulator is not None:
                layer_name = f"model.layers.{layer_idx}.self_attn"
                if layer_name in self.activation_accumulator.cov_matrices:
                    try:
                        cov = self.activation_accumulator.get_covariance(layer_name)
                    except ValueError:
                        cov = None

            q_nope_2d = W_q_nope_all[layer_idx].reshape(-1, W_q_nope_all.shape[-1])
            q_rope_2d = W_q_rope_all[layer_idx].reshape(-1, W_q_rope_all.shape[-1])
            k_nope_2d = W_k_nope_all[layer_idx].reshape(-1, W_k_nope_all.shape[-1])
            k_rope_single = W_k_rope_all[layer_idx]
            if k_rope_single.ndim == 3:
                k_rope_single = k_rope_single.mean(dim=0)
            v_2d = W_v_all[layer_idx].reshape(-1, W_v_all.shape[-1])

            if self.model is not None and hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                o_weight = self.model.model.layers[layer_idx].self_attn.o_proj.weight
            else:
                o_weight = torch.randn(q_nope_2d.shape[-1], v_2d.shape[0], dtype=q_nope_2d.dtype, device=q_nope_2d.device)

            layer_weights = vllm_converter.convert_decoupled_layer_to_vllm(
                w_q_nope=q_nope_2d,
                w_q_rope=q_rope_2d,
                w_k_nope=k_nope_2d,
                w_k_rope=k_rope_single,
                w_v=v_2d,
                w_o=o_weight,
                cov_matrix=cov,
            )

            for name, tensor in layer_weights.items():
                vllm_state_dict[f"model.layers.{layer_idx}.self_attn.{name}"] = tensor

        torch.save(vllm_state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        vllm_converter.generate_vllm_config(output_dir, num_experts=num_experts, top_k=top_k)
        print(f"Model saved to {output_dir} and ready for vLLM.")

    def _first_nonempty_text(self, min_chars: int = 32):
        """WikiText often starts with blank rows; skip those for evaluation."""
        for row in self.data:
            text = row.get("text") or row.get("content") or ""
            if isinstance(text, str) and len(text.strip()) >= min_chars:
                return text
        raise ValueError(
            f"No non-empty text sample found in dataset "
            f"(looked for >= {min_chars} chars)."
        )

    def evaluate(self):
        try:
            device = next(self.model.parameters()).device
            evaluation = self.Evaluation(self.model, self.data, str(device))
            text = self._first_nonempty_text()
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.mla_config.max_seq_len,
            ).to(device)
            if inputs["input_ids"].numel() < 2:
                raise ValueError(
                    "Tokenized evaluation sample is too short for perplexity "
                    f"(got {inputs['input_ids'].numel()} tokens)."
                )
            evaluation.evaluate(inputs)
        except Exception as e:
            print(f"Error during evaluation: {e}")
