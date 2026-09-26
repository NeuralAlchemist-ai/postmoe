import torch
import torch.nn as nn
from dataclasses import replace

from postmoe.eval import Evaluation
from postmoe.extractor import UniversalAttentionDecoupler
from postmoe.converter import Converter
from postmoe.MLA import MLA_Attention

from loader.load_model import ModelLoader
from loader.data_loader import DataLoader


def _make_linear(weight_2d):
    """Wrap a 2D weight tensor [out_features, in_features] into nn.Linear(bias=False)."""
    out_f, in_f = weight_2d.shape
    lin = nn.Linear(in_f, out_f, bias=False)
    lin.weight = nn.Parameter(weight_2d)
    return lin


class Pipeline:
    def __init__(self, config):
        self.UniversalAttentionDecoupler = UniversalAttentionDecoupler
        self.Converter = Converter
        self.Evaluation = Evaluation
        self.MLA_Attention = MLA_Attention

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
        self.all_weights = None            # extracted per-layer weights (tuple of 4D tensors)

    def convert(self):
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

        # Extract and decouple weights for ALL layers at once
        decoupler = self.UniversalAttentionDecoupler(
            self.model, self.hf_config, self.rope_dim
        )
        self.all_weights = decoupler.convert_to_target_layout(
            repeat_mode="gqa_broadcast"
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
        #   k_rope_i:  [num_q_heads, rope_dim, hidden]
        #   q_nope_i:  [num_q_heads, nope_dim, hidden]
        #   q_rope_i:  [num_q_heads, rope_dim, hidden]
        #   v_i:       [num_kv_heads, head_dim, hidden]
        k_nope_i = W_k_nope_all[layer_idx]
        k_rope_i = W_k_rope_all[layer_idx]
        q_nope_i = W_q_nope_all[layer_idx]
        q_rope_i = W_q_rope_all[layer_idx]
        v_i      = W_v_all[layer_idx]

        num_q_heads = q_nope_i.shape[0]
        hidden = q_nope_i.shape[-1]

        # ── Expand V to match num_q_heads (same broadcast as K) ──
        gqa_group = num_q_heads // v_i.shape[0]
        v_expanded = v_i.repeat_interleave(gqa_group, dim=0)   # [num_q_heads, head_dim, hidden]

        # ── Flatten heads → 2D for SVD ──
        k_nope_2d = k_nope_i.reshape(-1, hidden)   # [num_q_heads * nope_dim, hidden]
        v_2d      = v_expanded.reshape(-1, hidden)  # [num_q_heads * head_dim, hidden]

        # ── SVD compress K_nope → (k_up_proj, kv_down_proj) ──
        k_up, kv_down = self.Converter(k_nope_2d).svd_compress()
        rank = kv_down.shape[0]   # the latent / compressed KV dimension

        # ── SVD compress V with SAME rank (MLA shares the latent space) ──
        v_up, _ = self.Converter(v_2d).svd_compress(target_rank=rank)

        # ── For MLA, k_rope is a single shared projection (mean across Q-broadcast heads) ──
        k_rope_single = k_rope_i.mean(dim=0)   # [rope_dim, hidden]

        # ── Flatten Q heads → 2D ──
        q_nope_2d = q_nope_i.reshape(-1, hidden)   # [num_q_heads * nope_dim, hidden]
        q_rope_2d = q_rope_i.reshape(-1, hidden)   # [num_q_heads * rope_dim, hidden]

        # ── Build per-layer config with actual kv_latent_dim from SVD ──
        layer_config = replace(self.mla_config, kv_latent_dim=rank)

        # ── Wrap in nn.Linear and build MLA_Attention ──
        return self.MLA_Attention(
            config=layer_config,
            kv_down_proj=_make_linear(kv_down),      # hidden → latent  (callable)
            k_up_proj=_make_linear(k_up),             # latent → H*nope  (accessed via .weight)
            v_up_proj=_make_linear(v_up),             # latent → H*head  (accessed via .weight)
            q_nope=_make_linear(q_nope_2d),           # hidden → H*nope  (callable)
            q_rope=_make_linear(q_rope_2d),           # hidden → H*rope  (callable)
            k_rope=_make_linear(k_rope_single),       # hidden → rope    (callable)
            o_proj=o_proj,                             # original (already nn.Linear)
        )

    def evaluate(self):
        try:
            evaluation = self.Evaluation(self.model, self.data, "cuda")
            inputs = self.tokenizer(
                self.data[0]["text"],
                return_tensors="pt",
                truncation=True,
                max_length=self.mla_config.max_seq_len,
            ).to("cuda")
            evaluation.evaluate(inputs)
        except Exception as e:
            print(f"Error during evaluation: {e}")
