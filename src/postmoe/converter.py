import math

import torch


# FlashInfer MLA Tensor-Core tiles need HEAD_DIM_CKV divisible by this.
# Rank 19 (auto energy) triggers "zero-sized o_frag" at JIT compile time.
FLASHINFER_CKV_ALIGN = 32


class Converter:
    def __init__(self, nope):
        self.nope = nope

    def _optimized_rank(self, U, threshold=0.9):
        """
        Determine the optimal rank for compression based on the cumulative energy of singular values.
        """
        cumulative_energy = torch.cumsum(U ** 2, dim=0)
        total_energy = cumulative_energy[-1]
        rank = torch.searchsorted(cumulative_energy, threshold * total_energy).item() + 1
        return rank

    @staticmethod
    def align_rank(rank: int, max_rank: int, multiple: int = FLASHINFER_CKV_ALIGN) -> int:
        """Round rank up to an MMA-friendly multiple, without exceeding max_rank."""
        if multiple <= 1:
            return min(rank, max_rank)
        aligned = int(math.ceil(rank / multiple) * multiple)
        aligned = max(multiple, aligned)  # never go below one tile
        return min(aligned, max_rank)

    def svd_compress(self, target_rank=None, align_multiple: int = FLASHINFER_CKV_ALIGN):
        """
        Perform SVD on the NOPE matrix and compress it by keeping only the top singular values.

        Args:
            target_rank: If provided, force this rank instead of auto-detecting.
                         Useful when K and V must share the same latent dimension.
            align_multiple: Round the chosen rank up to a multiple of this value so
                            FlashInfer MLA kernels can compile (HEAD_DIM_CKV).
        """
        self.nope = self.nope.detach().to(torch.float32)
        U, S, Vh = torch.linalg.svd(self.nope, full_matrices=False)
        max_rank = len(S)

        if target_rank is None:
            rank = self._optimized_rank(S)
        else:
            rank = min(int(target_rank), max_rank)

        rank = self.align_rank(rank, max_rank, multiple=align_multiple)

        up = U[:, :rank] @ torch.diag(S[:rank])
        down = Vh[:rank, :]
        return up, down
