import torch

class Converter:
    def __init__(self, nope):
        self.nope = nope

    def _optimized_rank(self, S, threshold=0.95):
        """
        Определяет оптимальный ранг сжатия на основе сингулярных чисел.
        Порог увеличен до 0.95 для лучшего сохранения точности модели.
        """
        cumulative_energy = torch.cumsum(S ** 2, dim=0)
        total_energy = cumulative_energy[-1]
        rank = torch.searchsorted(cumulative_energy, threshold * total_energy).item() + 1
        return rank

    def svd_compress(self, target_rank=None):
        """
        Выполняет сбалансированное SVD-сжатие матрицы NOPE для конвертации в MLA.
        
        Args:
            target_rank: Если передан, принудительно устанавливает этот ранг.
                         Полезно для явного контроля сжатия KV-кэша.
        """
        original_dtype = self.nope.dtype
        # SVD вычисления обязательно проводим в float32 для стабильности
        self.nope = self.nope.detach().to(torch.float32)
        
        U, S, Vh = torch.linalg.svd(self.nope, full_matrices=False)
        max_rank = len(S)

        if target_rank is None:
            rank = self._optimized_rank(S)
        else:
            rank = min(int(target_rank), max_rank)

        # РЕШЕНИЕ ПРОБЛЕМЫ ПЕРПЛЕКСИИ: 
        # Распределяем сингулярные числа симметрично через квадратный корень.
        # Это сохраняет дисперсию весов и предотвращает взрыв активаций.
        S_sqrt = torch.diag(torch.sqrt(S[:rank]))
        
        up = U[:, :rank] @ S_sqrt
        down = S_sqrt @ Vh[:rank, :]
        
        # Возвращаем тензоры в исходном типе данных модели (BF16 или FP16)
        return up.to(original_dtype), down.to(original_dtype)
