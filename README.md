# postmoe

Convert Hugging Face **GQA** (Grouped-Query Attention) models into **MLA** (Multi-head Latent Attention) by decoupling Q/K into RoPE and NoPE parts, SVD-compressing the NoPE + V weights into a shared KV latent space, and swapping each layer’s attention for a FlashInfer-backed MLA module.

Requires **Python ≥ 3.11**, CUDA, and the dependencies in `pyproject.toml` (`torch`, `transformers`, `datasets`, `flashinfer-python`, `pytest`).

## Quick start

```bash
uv sync
uv run python src/run_pipeline.py \
  --model_name Qwen/Qwen2-0.5B \
  --rope_dim 32 \
  --dataset_name Salesforce/wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --split "test[:100]"
```

```bash
uv run pytest
```

---

## Pipeline flow

```
run_pipeline.py
  → Config
  → Pipeline.convert()
       ModelLoader  → model + tokenizer + HF config
       DataLoader   → eval dataset
       UniversalAttentionDecoupler → per-layer W_k/W_q (nope/rope) + W_v
  → for each layer: custom_MLA_attention()
       Converter.svd_compress() → kv_down / k_up / v_up
       MLA_Attention(...) replaces layer.self_attn
  → Pipeline.evaluate()
```

---

## File reference

### Root

| File | Role |
|------|------|
| `pyproject.toml` | Package metadata, runtime deps, and pytest `pythonpath = ["src"]`. |
| `uv.lock` | Locked dependency versions for `uv`. |
| `.python-version` | Pins Python `3.11`. |
| `LICENSE` | MIT. |
| `README.md` | This file. |

---

### `src/run_pipeline.py` — CLI entrypoint

Parses CLI args, builds a `Config`, runs the conversion, replaces every layer’s `self_attn` with MLA, then evaluates.

| Function / block | How it works |
|------------------|--------------|
| `argparse` setup | Exposes `--model_name`, `--rope_dim`, `--dataset_name`, `--dataset_config`, `--split` (defaults match `Config`). |
| Main body | Instantiates `Pipeline(config)`, calls `convert()`, then for each `pipeline.model.model.layers[i]` sets `layer.self_attn = pipeline.custom_MLA_attention(i, layer.self_attn.o_proj)` (reuses the original output projection), then `evaluate()`. |

---

### `src/pipeline.py` — orchestration

#### `_make_linear(weight_2d)`

Wraps a 2D weight `[out_features, in_features]` in `nn.Linear(..., bias=False)` and assigns it as a `Parameter`. Used so SVD factors and extracted Q/K projections become callable modules for `MLA_Attention`.

#### `Pipeline`

| Method | How it works |
|--------|--------------|
| `__init__(config)` | Stores class refs (`UniversalAttentionDecoupler`, `Converter`, `Evaluation`, `MLA_Attention`, loaders) and config fields (`model_name`, `rope_dim`, MLA dims, dataset settings). Holds runtime state: `model`, `tokenizer`, `hf_config`, `data`, `all_weights`. |
| `convert()` | Loads model/tokenizer/HF config via `ModelLoader`; loads the dataset via `DataLoader`; builds `UniversalAttentionDecoupler(model, hf_config, rope_dim)` and sets `all_weights` from `convert_to_target_layout(repeat_mode="gqa_broadcast")` — a 5-tuple of stacked tensors for all layers. |
| `custom_MLA_attention(layer_idx, o_proj)` | Indexes one layer’s weights from `all_weights`. Broadcasts V from KV heads to Q heads (`repeat_interleave` by GQA group size). Flattens K_nope and V to 2D, SVD-compresses K_nope to `(k_up, kv_down)`, then compresses V to the **same** rank. Averages K_rope over broadcast Q heads into one shared projection. Builds a per-layer `Config` with `kv_latent_dim=rank` and returns `MLA_Attention` with linear wrappers for `kv_down`, `k_up`, `v_up`, `q_nope`, `q_rope`, `k_rope`, plus the original `o_proj`. |
| `evaluate()` | Tokenizes the first dataset sample (truncated to `max_seq_len`), runs `Evaluation(...).evaluate(inputs)` on CUDA, prints errors if anything fails. |

---

### `src/postmoe/config.py` — MLA hyperparameters

#### `Config` (dataclass)

| Field | Meaning |
|-------|---------|
| `model_name` | Hugging Face model id (default `Qwen/Qwen2-0.5B`). |
| `dataset_*` / `split` | Hugging Face dataset load settings. |
| `n_head` | Number of attention heads for MLA. |
| `kv_latent_dim` | Compressed KV latent size (overridden per layer by SVD rank in the pipeline). |
| `nope_dim` | Non-RoPE head dimension. |
| `rope_dim` | RoPE head dimension (must match how weights are split). |
| `head_dim` | Per-head value / output dim (often equals `nope_dim`). |
| `max_seq_len` | RoPE cache length and tokenizer truncation bound. |

---

### `src/postmoe/extractor.py` — GQA → NoPE/RoPE weight split

#### `UniversalAttentionDecoupler`

Reads every layer’s `q_proj` / `k_proj` / `v_proj`, reshapes into heads, splits Q/K into RoPE and NoPE slices, and optionally broadcasts KV-head K weights to Q-head count for MLA.

| Method | How it works |
|--------|--------------|
| `__init__(model, config, rope_dim_per_head, split_strategy="suffix")` | Reads `hidden_size`, `num_key_value_heads`, `num_attention_heads`, `num_hidden_layers` from the HF config; derives `gqa_group_size` and `total_head_dim` (`hidden_size // num_q_heads`, or `kv_channels` if present). |
| `_universal_qkv_proj_extractor()` | Loops layers, detaches `k_proj` / `q_proj` / `v_proj` weights, stacks to `[num_layers, out, in]`. |
| `_extract_weights()` | Views stacks as `[layers, heads, head_dim, hidden]`. Splits Q/K along the head-dim axis by `split_strategy`: **suffix** (last `rope_dim`), **prefix** (first `rope_dim`), or **interleaved** (every `head_dim // rope_dim`-th index). Returns `(W_k_nope, W_k_rope, W_q_nope, W_q_rope, W_v)`. |
| `convert_to_target_layout(repeat_mode="gqa_broadcast")` | If `repeat_mode` is `None`, returns extracted tensors unchanged. For `"gqa_broadcast"`, `repeat_interleave`s K_nope and K_rope along the head axis by `gqa_group_size` so KV-head K matches Q-head count (GQA → MLA layout). V stays at KV-head count. |

---

### `src/postmoe/converter.py` — SVD compression

#### `Converter`

| Method | How it works |
|--------|--------------|
| `__init__(nope)` | Stores the 2D weight matrix to compress (`[out, in]`). |
| `_optimized_rank(U, threshold=0.9)` | Given singular values `S` (named `U` in the signature), computes cumulative energy of `S²` and returns the smallest rank retaining `threshold` (default 90%) of total energy. |
| `svd_compress(target_rank=None, align_multiple=32)` | Casts to float32, runs `torch.linalg.svd(...)`. Rank is auto via `_optimized_rank(S)` or forced, then **rounded up** to an MMA multiple (default 32) so FlashInfer `HEAD_DIM_CKV` can JIT. Returns `up` / `down` with `up @ down ≈ original`. |
| `align_rank(rank, max_rank, multiple=32)` | `ceil(rank / multiple) * multiple`, clamped to `max_rank`. |

---

### `src/postmoe/MLA.py` — FlashInfer MLA attention

#### `MLA_Attention`

Drop-in attention module: projects into latent KV + RoPE, runs FlashInfer batch MLA, absorbs up-projections, applies `o_proj`.

| Method | How it works |
|--------|--------------|
| `__init__(...)` | Stores projections; allocates a 128 MB CUDA workspace and `BatchMLAPagedAttentionWrapper`; builds and registers non-persistent `cos_cache` / `sin_cache` for RoPE. |
| `_build_rope_cache(max_seq_len, rope_dim, theta=10000)` | Classic RoPE inverse frequencies; outer product with positions; concatenates freqs with themselves (duplicated layout for half-split rotation); returns `(cos, sin)`. |
| `freq_rope(seq_len, device, past_seq_len=0)` | Slices caches for `[past_seq_len, past_seq_len + seq_len)`; errors if past max length. |
| `apply_rope(x, cos, sin)` | Half-split rotate: `x1, x2 = split(x)`; returns `x * cos + cat(-x2, x1) * sin`. |
| `forward(hidden_states, past_seq_len, ...)` | Projects `q_nope`, `q_rope`, latent `c_kv` (`kv_down_proj`), and shared `k_rope`; applies RoPE to Q/K rope parts. Absorbs `k_up` into queries via einsum (`q_nope` × `W_UK` → latent queries). Plans FlashInfer MLA with page size 1 and runs it; absorbs `v_up` on the latent output; returns `(o_proj(out), (ckv_cache, kpe_cache))`. Softmax scale is `1 / sqrt(nope_dim + rope_dim)`. |

---

### `src/postmoe/eval.py` — metrics

#### `Evaluation`

| Method | How it works |
|--------|--------------|
| `__init__(model, dataset, device)` | Stores model, dataset, and device string. |
| `_kv_cache(base_vram)` | Estimates KV-related VRAM as peak allocated minus baseline; prints MB. |
| `_perplexity(logits, inputs)` | Next-token CE loss on shifted logits/labels; prints `exp(loss)`. |
| `evaluate(inputs)` | Resets peak memory, records baseline VRAM, runs `model.generate(..., max_new_tokens=128, use_cache=False)` under `no_grad`, then reports KV estimate and perplexity from `outputs.logits`. |

---

### `src/loader/load_model.py` — model loading

#### `ModelLoader`

| Method | How it works |
|--------|--------------|
| `__init__(model_name)` | Stores id; initializes `tokenizer` / `model` / `config` to `None`. |
| `load_model()` | `AutoTokenizer` + `AutoModelForCausalLM.from_pretrained`; sets `config = model.config`; prints success or error. |
| `get_tokenizer()` / `get_model()` / `get_config()` | Return loaded objects or raise if `load_model()` was not called. |

---

### `src/loader/data_loader.py` — dataset loading

#### `DataLoader`

| Method | How it works |
|--------|--------------|
| `__init__(dataset_name, dataset_config, split)` | Stores HF dataset args. |
| `load_data()` | Calls `datasets.load_dataset(...)`; wraps failures in `RuntimeError` with name/config/split; returns the dataset. |
| `get_data()` | Returns cached data or raises if not loaded. |

---

### Package init stubs

- `src/postmoe/__init__.py` — empty package marker.
- `src/loader/__init__.py` — empty package marker.

---

## Tests (`tests/`)

| File | What it covers |
|------|----------------|
| `test_extractor.py` | Dummy GQA model: suffix/prefix/interleaved splits, `repeat_mode=None` vs broadcast, invalid modes. |
| `test_converting_math.py` | SVD auto-rank, forced rank, rank clamping, energy-based `_optimized_rank`, exact low-rank reconstruction. |
| `test_mla.py` | RoPE cache / `freq_rope`, `apply_rope` math, CUDA forward of `MLA_Attention`. |
| `test_pipeline.py` | `_make_linear` shape/bias; `custom_MLA_attention` construction with mocked weights. |
| `test_eval.py` | Perplexity and KV VRAM helper printouts (no full generate). |
| `test_system.py` | End-to-end-ish GQA→MLA conversion path. |

---

## Notes

- Conversion assumes a Transformers decoder layout: `model.model.layers[i].self_attn` with `q_proj` / `k_proj` / `v_proj` / `o_proj`.
- `rope_dim` must match how the source model packs RoPE vs content dims in each head.
- MLA forward expects CUDA (FlashInfer workspace and kernels).
- Per-layer `kv_latent_dim` comes from SVD on that layer’s K_nope, not only from the default in `Config`.
