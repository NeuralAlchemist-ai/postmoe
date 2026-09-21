from dataclasses import dataclass

@dataclass 
class User_Config:
    model_name: str = "facebook/opt-1.3b"
    rope_dim: int = 32
    data_path: str = "data/processed_data.json"

