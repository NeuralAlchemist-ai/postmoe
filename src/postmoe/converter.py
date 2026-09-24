import torch


class Converter:
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
        self.nope = self.nope.detach().to(torch.float32)
        U, S, Vh = torch.linalg.svd(self.nope, full_matrices=False)

        # Determine the rank for compression (you can adjust this)
        rank = self._optimized_rank(S)

        # Compress the matrices by keeping only the top singular values
        up = U[:, :rank] @ torch.diag(S[:rank])
        down = Vh[:rank, :]
        return up, down
        