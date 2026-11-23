"""molt vs. BlinkDL's official model code, every supported generation, float32.

Needs CUDA and the checkpoints; set MOLT_TEST_MODELS to the folder holding them.
"""
import os
from pathlib import Path

import pytest
import torch

from molt import RWKV7
from official_harness import load_official

MODELS = Path(os.environ.get("MOLT_TEST_MODELS", "D:/models/rwkv"))
CASES = {
    "x070": "rwkv7-g1d-0.1b-20260129-ctx8192.pth",
    "x070a": "rwkv7a-g1d-0.1b-20260212-ctx8192.pth",
    "x070b": "rwkv7b-g1b-0.1b-20250822-ctx4096.pth",
    "x080": "rwkv-8-pile-rc00-20250508.pth",
}

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("variant", list(CASES))
def test_parity_with_official(variant):
    path = MODELS / CASES[variant]
    if not path.exists():
        pytest.skip(f"missing {path}")
    ours = RWKV7.from_pretrained(path, dtype=torch.float32)
    ours.backend = "torch"
    c = ours.cfg
    assert c.variant == variant
    ref = load_official(variant, path, c.n_layer, c.n_embd, c.head_size, c.vocab_size)

    torch.manual_seed(0)
    ids = torch.randint(1, c.vocab_size, (40,)).tolist()
    split = 25
    with torch.no_grad():
        # official: two calls, so the second one continues from the returned state (and, for x070b, the KV cache)
        r1, st = ref.forward(ids[:split], None, full_output=True)
        r2, st = ref.forward(ids[split:], st, full_output=True)
        ref_logits = torch.cat([r1, r2]).float()
        # ours: one parallel call, and the same split with state carried over
        full, _ = ours(torch.tensor([ids], device="cuda"))
        a, s = ours(torch.tensor([ids[:split]], device="cuda"))
        b, _ = ours(torch.tensor([ids[split:]], device="cuda"), s)
    full, chunked = full[0].float(), torch.cat([a[0], b[0]]).float()
    scale = ref_logits.abs().max().item()
    assert (full - ref_logits).abs().max().item() < 1e-3 * scale + 1e-3
    assert (chunked - ref_logits).abs().max().item() < 1e-3 * scale + 1e-3
