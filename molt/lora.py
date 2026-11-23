"""LoRA adapters: the slow, parametric memory tier.

The frozen base stays in bf16; adapters are float32 so that tiny online updates are not
rounded away (a bf16 weight cannot move by less than ~1/256 of its magnitude).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("att.receptance", "att.key", "att.value", "att.output", "ffn.key", "ffn.value")


class LoRALinear(nn.Module):
    enabled = True  # class-wide switch, see lora_disabled()

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # dial: how much of the learned update to apply (0 = before learning, 1 = as learned, >1 = amplified).
        # A tensor changed in place, so CUDA-graph decoding sees it.
        self.register_buffer("dial", torch.ones((), device=dev), persistent=False)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        y = self.base(x)
        if not LoRALinear.enabled:
            return y
        return y + ((x.float() @ self.lora_A.t()) @ self.lora_B.t() * (self.scale * self.dial)).to(y.dtype)

    @torch.no_grad()
    def merged_weight(self):
        return (self.base.weight.float() + self.scale * self.lora_B @ self.lora_A).to(self.base.weight.dtype)


class lora_disabled:
    """with lora_disabled(): ... runs the frozen base model (used as the anchor for KL)."""

    def __enter__(self):
        LoRALinear.enabled = False

    def __exit__(self, *exc):
        LoRALinear.enabled = True


@torch.no_grad()
def set_dial(model, value: float, layers=None):
    """Scale the learned LoRA update globally or for some layers (list of layer ids)."""
    for n, m in model.named_modules():
        if isinstance(m, LoRALinear) and (layers is None or int(n.split(".")[1]) in layers):
            m.dial.fill_(value)


@torch.no_grad()
def lora_layer_norms(model) -> list[float]:
    """Per-layer size of the applied LoRA update, sqrt(sum over modules ||dial*scale*B@A||_F^2).
    Uses ||BA||_F^2 = tr((B^T B)(A A^T)), r x r matrices only, so it is cheap enough for every step."""
    out = [0.0] * model.cfg.n_layer
    for n, m in model.named_modules():
        if isinstance(m, LoRALinear):
            sq = torch.trace((m.lora_B.t() @ m.lora_B) @ (m.lora_A @ m.lora_A.t())) * (m.scale * m.dial) ** 2
            out[int(n.split(".")[1])] += float(sq)
    return [round(x ** 0.5, 5) for x in out]


def add_lora(model, r: int = 16, alpha: float = 32.0, targets=DEFAULT_TARGETS) -> list[str]:
    """Wrap matching Linear layers in place; returns the wrapped module names."""
    done = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(full.endswith(t) for t in targets):
                setattr(mod, child_name, LoRALinear(child, r, alpha))
                done.append(full)
    return done


def lora_state_dict(model) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def merged_state_dict(model) -> dict:
    """Official-format state dict with adapters folded in (loadable by any RWKV-7 tool)."""
    out = {}
    lora_mods = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    for k, v in model.state_dict().items():
        if "lora_" in k:
            continue
        if ".base." in k:
            mod_name = k.rsplit(".base.", 1)[0]
            k = k.replace(".base.", ".")
            if k.endswith("weight"):
                v = lora_mods[mod_name].merged_weight()
        out[k] = v.detach().cpu()
    return out
