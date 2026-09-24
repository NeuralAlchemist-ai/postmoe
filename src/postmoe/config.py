from dataclasses import dataclass

@dataclass 
class Config:
    model_name: str = "Qwen/Qwen2-0.5B"
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "test[:100]"

    n_head: int = 16
    kv_latent_dim: int = 64
    nope_dim: int = 64
    rope_dim: int = 32
