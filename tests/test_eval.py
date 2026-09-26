import torch
from postmoe.eval import Evaluation


def test_eval_perplexity_calculation(capsys):
    evaluation = Evaluation(model=None, dataset=None, device="cpu")

    batch_size = 2
    seq_len = 10
    vocab_size = 50

    logits = torch.randn(batch_size, seq_len, vocab_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    inputs = {"input_ids": input_ids}

    # Should compute perplexity and print it
    evaluation._perplexity(logits, inputs)
    captured = capsys.readouterr()
    assert "Perplexity:" in captured.out


def test_eval_kv_cache_vram(capsys):
    evaluation = Evaluation(model=None, dataset=None, device="cpu")
    # Calling _kv_cache with a dummy base_vram value
    evaluation._kv_cache(base_vram=0)
    captured = capsys.readouterr()
    assert "Estimated KV Cache VRAM Usage:" in captured.out
