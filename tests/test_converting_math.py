import torch
import pytest
from postmoe.converter import Converter


def test_svd_compress_auto_rank():
    """Test SVD compression with automatic rank selection."""
    matrix = torch.randn(128, 64)
    converter = Converter(matrix)
    up, down = converter.svd_compress()

    assert up is not None and down is not None, "Outputs should not be None."
    assert isinstance(up, torch.Tensor) and isinstance(down, torch.Tensor), "Outputs must be PyTorch tensors."

    # up is [M, rank], down is [rank, N]
    assert up.shape[0] == 128, f"Expected up rows to be 128, got {up.shape[0]}"
    assert down.shape[1] == 64, f"Expected down cols to be 64, got {down.shape[1]}"
    assert up.shape[1] == down.shape[0], f"Rank mismatch: {up.shape[1]} vs {down.shape[0]}"

    # Reconstruction shape should match original matrix shape
    reconstructed = up @ down
    assert reconstructed.shape == (128, 64), f"Reconstructed shape {reconstructed.shape} != (128, 64)"

    assert torch.all(torch.isfinite(up)), "up matrix should not contain NaN or Inf."
    assert torch.all(torch.isfinite(down)), "down matrix should not contain NaN or Inf."


def test_svd_compress_target_rank():
    """Test SVD compression with explicitly specified target rank."""
    matrix = torch.randn(100, 80)
    target_rank = 16
    converter = Converter(matrix)
    up, down = converter.svd_compress(target_rank=target_rank)

    assert up.shape == (100, target_rank), f"Expected up shape (100, {target_rank}), got {up.shape}"
    assert down.shape == (target_rank, 80), f"Expected down shape ({target_rank}, 80), got {down.shape}"
    assert (up @ down).shape == (100, 80)


def test_svd_compress_target_rank_clamping():
    """Test that target rank does not exceed maximum possible singular values."""
    matrix = torch.randn(20, 15)  # Max rank is 15
    converter = Converter(matrix)
    up, down = converter.svd_compress(target_rank=50)

    assert up.shape[1] == 15, f"Expected rank clamped to 15, got {up.shape[1]}"
    assert down.shape[0] == 15, f"Expected rank clamped to 15, got {down.shape[0]}"


def test_optimized_rank():
    """Test rank determination based on energy threshold."""
    # Singular values where first 2 contain ~95% of energy: 10^2 + 5^2 = 125, 1^2 + 1^2 = 2. Total = 127
    S = torch.tensor([10.0, 5.0, 1.0, 1.0])
    converter = Converter(torch.eye(4))

    rank = converter._optimized_rank(S, threshold=0.9)
    assert rank == 2, f"Expected rank 2 for 90% energy, got {rank}"

    rank_high = converter._optimized_rank(S, threshold=0.99)
    assert rank_high >= 3, f"Expected rank >= 3 for 99% energy, got {rank_high}"


def test_svd_compress_exact_low_rank():
    """Test SVD compression on an exact low-rank matrix."""
    # Construct exact rank-4 matrix: A = B @ C
    B = torch.randn(50, 4)
    C = torch.randn(4, 30)
    exact_low_rank = B @ C

    converter = Converter(exact_low_rank)
    up, down = converter.svd_compress(target_rank=4)

    reconstructed = up @ down
    assert torch.allclose(exact_low_rank, reconstructed, atol=1e-4), "Rank-4 matrix should be reconstructed accurately."
