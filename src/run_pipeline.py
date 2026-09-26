import argparse

from pipeline import Pipeline
from postmoe.config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B")
parser.add_argument("--rope_dim", type=int, default=32)
parser.add_argument("--dataset_name", type=str, default="Salesforce/wikitext")
parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
parser.add_argument("--split", type=str, default="test[:100]")
args = parser.parse_args()

config = Config(
    model_name=args.model_name,
    rope_dim=args.rope_dim,
    dataset_name=args.dataset_name,
    dataset_config=args.dataset_config,
    split=args.split,
)

pipeline = Pipeline(config)
pipeline.convert()

for i, layer in enumerate(pipeline.model.model.layers):
    layer.self_attn = pipeline.custom_MLA_attention(i, layer.self_attn.o_proj)

pipeline.evaluate()