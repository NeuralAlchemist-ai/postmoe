
import torch

class Evaluation:
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device

    def _kv_cache(self, base_vram):
        
        peak_vram = torch.cuda.max_memory_allocated()
        kv_cache_vram = peak_vram - base_vram

        print(f"Estimated KV Cache VRAM Usage: {kv_cache_vram / 1024**2:.2f} MB")


    def _perplexity(self, logits, inputs):
    
        # 2. Calculate perplexity
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs['input_ids'][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = torch.exp(loss)

        print(f"Perplexity: {perplexity.item():.4f}")

    def evaluate(self, inputs):
        torch.cuda.reset_peak_memory_stats()
        base_vram = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=128, use_cache=False)
            logits = outputs.logits
            
        self._kv_cache(base_vram)
        self._perplexity(logits, inputs)