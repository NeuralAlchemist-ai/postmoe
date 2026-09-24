import argparse

from torchgen import model
from pipeline import Pipeline
from postmoe.config import User_Config

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt2")
parser.add_argument("--rope_dim", type=int, default=64)
parser.add_argument("--data_path", type=str, default="data.txt")
parser.add_argument("--data_config", type=str, default="wikitext-2-raw-v1")  # Added argument for dataset configuration
args = parser.parse_args()

config = User_Config(
    model_name=args.model_name,
    rope_dim=args.rope_dim,
    data_path=args.data_path
)

pipeline = Pipeline(config)
pipeline.convert()

for layer in pipeline.model.model.layers:
    layer.self_attn = pipeline.custom_MLA_attention(layer.self_attn)

pipeline.evaluate()