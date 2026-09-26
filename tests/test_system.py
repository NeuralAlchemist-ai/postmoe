import os
import pytest
import torch
from pipeline import Pipeline
from postmoe.config import Config
from postmoe.MLA import MLA_Attention


def test_gqa_to_mla_converting():
    # Force offline mode for Hugging Face so cached data & weights are used immediately without network timeout
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = Config(
        model_name="Qwen/Qwen2-0.5B",
        rope_dim=32,
        dataset_name="Salesforce/wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="test[:10]",
    )

    pipeline = Pipeline(config)
    pipeline.convert()

    # Verify model, tokenizer, and config loading
    assert pipeline.model is not None, "Model should be loaded."
    assert pipeline.tokenizer is not None, "Tokenizer should be loaded."
    assert pipeline.hf_config is not None, "HF config should be loaded."
    assert pipeline.data is not None, "Dataset should be loaded."

    # Verify decoupled weights extraction
    assert pipeline.all_weights is not None, "all_weights should be extracted after conversion."
    W_k_nope, W_k_rope, W_q_nope, W_q_rope, W_v = pipeline.all_weights

    num_layers = len(pipeline.model.model.layers)
    assert W_k_nope.shape[0] == num_layers
    assert W_k_rope.shape[0] == num_layers
    assert W_q_nope.shape[0] == num_layers
    assert W_q_rope.shape[0] == num_layers
    assert W_v.shape[0] == num_layers

    # Test converting a layer's attention to custom MLA attention
    if torch.cuda.is_available():
        layer_idx = 0
        original_o_proj = pipeline.model.model.layers[layer_idx].self_attn.o_proj
        mla_layer = pipeline.custom_MLA_attention(layer_idx, original_o_proj)

        assert isinstance(mla_layer, MLA_Attention), "Converted layer must be an instance of MLA_Attention."
        assert mla_layer.o_proj is original_o_proj
        assert mla_layer.config.rope_dim == config.rope_dim

        # Verify replacement into model layer works as intended in run_pipeline.py
        pipeline.model.model.layers[layer_idx].self_attn = mla_layer
        assert pipeline.model.model.layers[layer_idx].self_attn is mla_layer