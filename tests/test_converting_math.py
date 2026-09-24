import torch
from postmoe.converter import Converter


def test_converting_math():
    converter = Converter(torch.randn(100, 100))  # Example NOPE matrix
    compressed_nope = converter.svd_compress()
    assert compressed_nope is not None, "Compressed NOPE matrix should not be None."
    assert compressed_nope.shape == (100, 100), "Compressed NOPE matrix should preserve the original shape."
    assert torch.all(torch.isfinite(compressed_nope)), "Compressed NOPE matrix should not contain NaN or Inf values."

