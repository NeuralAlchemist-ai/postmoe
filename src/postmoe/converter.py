import torch
import torch.nn as nn

class Converter:
    def __init__(self, nope):
        self.nope = nope

    def _optimized_rank(self, S, threshold=0.95):
        cumulative_energy = torch.cumsum(S ** 2, dim=0)
        total_energy = cumulative_energy[-1]
        rank = torch.searchsorted(cumulative_energy, threshold * total_energy).item() + 1
        return rank

    def svd_compress(self, target_rank=None):
        original_dtype = self.nope.dtype
        self.nope = self.nope.detach().to(torch.float32)
        U, S, Vh = torch.linalg.svd(self.nope, full_matrices=False)
        max_rank = len(S)

        if target_rank is None:
            rank = self._optimized_rank(S)
        else:
            rank = min(int(target_rank), max_rank)

        S_sqrt = torch.diag(torch.sqrt(S[:rank]))
        up = U[:, :rank] @ S_sqrt
        down = S_sqrt @ Vh[:rank, :]
        return up.to(original_dtype), down.to(original_dtype)


class LayerActivationAccumulator:
    def __init__(self, model: nn.Module, layers_to_track=None):
        """
        Args:
            model: Модель PyTorch.
            layers_to_track: Список типов слоев для отслеживания. По умолчанию только внимание (W_qkv или аналоги).
        """
        self.model = model  # ФИКС: Сохраняем модель
        self.cov_matrices = {}  
        self.total_tokens = {}  
        self.hooks = []
        
        # Если типы слоев не заданы, лучше не брать MLP, чтобы не ломать hidden_size
        self.layers_to_track = layers_to_track if layers_to_track else (nn.Linear,)
        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, self.layers_to_track):
                # ФИКС: Динамически определяем in_features для каждого слоя вместо фиксированного H
                in_features = module.in_features
                
                self.cov_matrices[name] = torch.zeros((in_features, in_features), dtype=torch.float32, device='cpu')
                self.total_tokens[name] = 0
                
                hook_fn = self._make_hook(name, in_features)
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)

    def _make_hook(self, layer_idx: str, in_features: int):
        def hook(module, input, output):
            # ФИКС: Делаем detach(), чтобы избежать утечек памяти
            input_tensor = input[0].detach() 
            
            # Извлекаем скрытую размерность текущего слоя
            X = input_tensor.view(-1, in_features)
            X = X.to(dtype=torch.float32)
            
            # Вычисляем XT * X на текущем девайсе (например, GPU)
            local_XTX = torch.mm(X.t(), X)
            
            # Аккумулируем на CPU
            device = self.cov_matrices[layer_idx].device
            self.cov_matrices[layer_idx].add_(local_XTX.to(device))
            self.total_tokens[layer_idx] += X.shape[0]
            
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def activation_collecting(self, dataloader: torch.utils.data.DataLoader):
        # ФИКС: Берем девайс модели динамически
        device = next(self.model.parameters()).device
        
        for batch in dataloader:
            # Безопасный перенос батча на GPU
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad():
                self.model(**batch)
        self.remove_hooks()

    def get_covariance(self, layer_idx: str) -> torch.Tensor:
        N = self.total_tokens[layer_idx]
        if N < 2:
            raise ValueError("Недостаточно токенов для вычисления ковариации.")
        # Возвращаем несмещенную uncentered матрицу моментов (обычно этого достаточно для SVD)
        return self.cov_matrices[layer_idx] / N


class DataDrivenSVD: 
    def __init__(self, W_k_nope: torch.Tensor, W_v: torch.Tensor):
        # W_k_nope и W_v имеют размерность [out_features, hidden_size]
        self.W_kv_stacked = torch.cat([W_k_nope, W_v], dim=0)
        self.out_k_dim = W_k_nope.shape[0]
        self.original_dtype = W_k_nope.dtype
        self.device = W_k_nope.device

    def _get_optimized_rank(self, S, threshold=0.98):
        cumulative_energy = torch.cumsum(S ** 2, dim=0)
        total_energy = cumulative_energy[-1]
        rank = torch.searchsorted(cumulative_energy, threshold * total_energy).item() + 1
        return rank

    def convert(self, target_rank=None, threshold=0.98, cov_matrix: torch.Tensor = None):
        W_kv_orig = self.W_kv_stacked.detach().to(torch.float32).to(self.device)
        
        if cov_matrix is not None:
            cov_matrix = cov_matrix.to(torch.float32).to(self.device)
            
            # 1. Вычисляем матричный корень и инверсию через eigh
            L, P = torch.linalg.eigh(cov_matrix)
            L = torch.clamp(L, min=1e-9) # Защита от деления на 0 при инверсии
            
            cov_sqrt = P @ torch.diag(torch.sqrt(L)) @ P.t()
            # Инверсия корня: (Sigma^{1/2})^{-1}
            cov_sqrt_inv = P @ torch.diag(1.0 / torch.sqrt(L)) @ P.t()
            
            # 2. Взвешиваем веса перед SVD
            W_kv_f32 = torch.matmul(W_kv_orig, cov_sqrt)
        else:
            W_kv_f32 = W_kv_orig
            cov_sqrt_inv = None

        # 3. Выполняем стандартное SVD
        U, S, Vh = torch.linalg.svd(W_kv_f32, full_matrices=False)
        
        max_rank = len(S)
        r = min(int(target_rank), max_rank) if target_rank is not None else min(self._get_optimized_rank(S, threshold), max_rank)
        S_sqrt = torch.sqrt(S[:r])
        
        # 4. Балансируем сингулярные числа
        W_up_total = U[:, :r] @ torch.diag(S_sqrt)          # [out_k + out_v, r]
        W_DKV = torch.diag(S_sqrt) @ Vh[:r, :]              # [r, hidden_size]
        
        # 5. Возвращаем веса в исходный базис активаций (компенсируем взвешивание)
        if cov_sqrt_inv is not None:
            W_DKV = torch.matmul(W_DKV, cov_sqrt_inv)
        
        # Разделяем обратно на К и V проекции
        W_UK = W_up_total[:self.out_k_dim, :]   
        W_UV = W_up_total[self.out_k_dim:, :]   
        
        return (
            W_DKV.to(self.original_dtype),
            W_UK.to(self.original_dtype),
            W_UV.to(self.original_dtype)
        )
