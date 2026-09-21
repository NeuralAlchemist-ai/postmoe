import torch


class converter:
    def __init__(self, nope):
        self.nope = nope

    def _optimized_rank(self, U, threshold=0.9):
        """
        Determine the optimal rank for compression based on the cumulative energy of singular values.
        """
        # Calculate the cumulative energy of singular values
        cumulative_energy = torch.cumsum(U ** 2, dim=0)
        total_energy = cumulative_energy[-1]

        # Determine the rank that retains 90% of the energy
        rank = torch.searchsorted(cumulative_energy, threshold * total_energy).item() + 1
        return rank
    
    def svd_compress(self):
        """
        Perform SVD on the NOPE matrix and compress it by keeping only the top singular values
        """
        # Perform SVD on the ROPE and NOPE matrices
        W_k_nope = self.nope.detach().to(torch.float32)
        U_nope, S_nope, Vh_nope = torch.linalg.svd(self.nope, full_matrices=False)

        # Determine the rank for compression (you can adjust this)
        rank_nope = self._optimized_rank(U_nope)

        # Compress the matrices by keeping only the top singular values
        compressed_nope = U_nope[:, :rank_nope] @ torch.diag(S_nope[:rank_nope]) @ Vh_nope[:rank_nope, :]

        return compressed_nope
        