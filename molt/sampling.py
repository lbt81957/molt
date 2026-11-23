from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 0.3
    top_k: int = 0
    presence_penalty: float = 0.5   # subtracted once a token has appeared
    frequency_penalty: float = 0.1  # subtracted per occurrence
    penalty_decay: float = 0.996    # occurrence counts fade, so long outputs are not over-penalised
    max_tokens: int = 512
    stop: tuple = ("\n\nUser:",)


def sample(logits: torch.Tensor, cfg: SamplingConfig, occurrence: torch.Tensor | None = None) -> int:
    """logits: [V] (any dtype). occurrence: [V] float32 running counts, updated in place."""
    logits = logits.float()
    if occurrence is not None:
        logits = logits - (occurrence > 0) * cfg.presence_penalty - occurrence * cfg.frequency_penalty
    if cfg.temperature <= 0:
        tok = int(logits.argmax())
    else:
        probs = torch.softmax(logits / cfg.temperature, dim=-1)
        if cfg.top_k > 0:
            kth = torch.topk(probs, cfg.top_k).values[-1]
            probs = torch.where(probs < kth, 0, probs)
        if 0 < cfg.top_p < 1:
            sp, _ = torch.sort(probs, descending=True)
            cutoff = sp[int((torch.cumsum(sp, 0) < cfg.top_p).sum())]
            probs = torch.where(probs < cutoff, 0, probs)
        tok = int(torch.multinomial(probs, 1))
    if occurrence is not None:
        occurrence.mul_(cfg.penalty_decay)
        occurrence[tok] += 1
    return tok
