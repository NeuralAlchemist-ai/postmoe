from src.pipeline import Pipeline
from src.postmoe.config import Config

def test_gqa_to_mla_converting():
    
    # Create a configuration for the test
    config = Config(
        model_name="Qwen/Qwen2-0.5B",
        rope_dim=32,
        dataset_name="Salesforce/wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="test[:10]",
    )

    # Initialize the pipeline with the test configuration
    pipeline = Pipeline(config)

    pipeline.convert()

    assert pipeline.model is not None
    assert pipeline.data is not None
    assert pipeline.compressed_nope is not None, "Compressed model should not be None after conversion."