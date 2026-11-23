"""Checkpoint-free tests: tokenizer, WKV backends (values and gradients), state I/O, LoRA."""
import pytest
import torch
import torch.nn.functional as F

from molt import RWKV7, RWKVConfig, RWKVTokenizer, State
from molt.lora import LoRALinear, add_lora, lora_disabled, merged_state_dict
from molt.ops import available_backends, wkv7

CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"


def test_tokenizer_roundtrip():
    tok = RWKVTokenizer()
    for s in ["Hello world", "你好，RWKV！\n\n", "emoji 🐍 and tabs\t", "User: hi\n\nAssistant:"]:
        assert tok.decode(tok.encode(s)) == s
    assert tok.encode("\n\n") == [261]


def _inputs(B=2, T=40, H=2, N=64, seed=0, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    r, k, v = (torch.randn(B, T, H, N, generator=g) * 0.5 for _ in range(3))
    kk = F.normalize(torch.randn(B, T, H, N, generator=g), dim=-1)
    a_gate = torch.rand(B, T, H, N, generator=g)
    w = -0.6065 * torch.rand(B, T, H, N, generator=g)
    s0 = torch.randn(B, H, N, N, generator=g) * 0.02
    xs = [t.to(DEV) for t in (r, w, k, v, -kk, kk * a_gate, s0)]
    if requires_grad:
        for t in xs:
            t.requires_grad_(True)
    return xs


def test_step_backend_matches_reference():
    r, w, k, v, a, b, s0 = _inputs(T=1)
    with torch.no_grad():
        o1, s1 = wkv7(r, w, k, v, a, b, s0, backend="torch")
        o2, s2 = wkv7(r, w, k, v, a, b, s0, backend="step")
    assert torch.allclose(o1, o2, atol=1e-5) and torch.allclose(s1, s2, atol=1e-5)


@pytest.mark.skipif(not (CUDA and "fla" in available_backends()), reason="needs CUDA + fla")
@pytest.mark.parametrize("backend", ["fla", "fla_rec"])
def test_fla_matches_reference_values_and_grads(backend):
    xs = _inputs(requires_grad=True)
    o_ref, s_ref = wkv7(*xs, backend="torch")
    (o_ref.square().mean() + s_ref.square().mean()).backward()
    g_ref = [t.grad.clone() for t in xs]
    for t in xs:
        t.grad = None
    o, s = wkv7(*(t.bfloat16() if i < 6 else t for i, t in enumerate(xs)), backend=backend)
    assert (o.float() - o_ref).abs().max() < 2e-2 * o_ref.abs().max()
    assert (s - s_ref).abs().max() < 2e-2 * s_ref.abs().max()
    if backend == "fla":
        (o.float().square().mean() + s.square().mean()).backward()
        for gr, t in zip(g_ref, xs):
            assert (t.grad.float() - gr).norm() < 0.1 * gr.norm()


def tiny_model(variant="x070"):
    cfg = RWKVConfig(vocab_size=65536, n_layer=2, n_embd=128, head_size=64, dim_ffn=256,
                     dim_w=16, dim_a=16, dim_v=16, dim_g=16, variant=variant)
    torch.manual_seed(0)
    m = RWKV7(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(torch.randn_like(p) * (0.02 if p.dim() > 1 else 0.1) + (1.0 if n.endswith("ln1.weight") or n.endswith("ln2.weight") else 0))
    return m.to(DEV).eval()


@pytest.mark.parametrize("variant", ["x070", "x070a", "x080"])
def test_parallel_equals_recurrent(variant):
    m = tiny_model(variant)
    m.backend = "torch"
    ids = torch.randint(0, 65536, (1, 12), device=DEV)
    with torch.no_grad():
        full, _ = m(ids)
        st, outs = None, []
        for t in range(12):
            lg, st = m(ids[:, t:t + 1], st)
            outs.append(lg[0, -1])
    assert torch.allclose(full[0], torch.stack(outs), atol=1e-4)


def test_state_save_load(tmp_path):
    m = tiny_model()
    _, s = m(torch.randint(0, 65536, (1, 5), device=DEV))
    s.save(tmp_path / "s.pth")
    s2 = State.load(tmp_path / "s.pth", DEV)
    assert torch.equal(s.wkv, s2.wkv) and s2.wkv.dtype == torch.float32


def test_lora_starts_as_identity_and_merges(tmp_path):
    m = tiny_model()
    ids = torch.randint(0, 65536, (1, 8), device=DEV)
    with torch.no_grad():
        base, _ = m(ids)
    names = add_lora(m, r=4)
    assert names and all(isinstance(m.get_submodule(n), LoRALinear) for n in names)
    with torch.no_grad():
        same, _ = m(ids)
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
        moved, _ = m(ids)
        with lora_disabled():
            off, _ = m(ids)
    assert torch.allclose(base, same) and torch.allclose(base, off) and not torch.allclose(base, moved)
    torch.save(merged_state_dict(m), tmp_path / "merged.pth")
    m2 = RWKV7.from_pretrained(tmp_path / "merged.pth", device=DEV, dtype=torch.float32)
    with torch.no_grad():
        again, _ = m2(ids)
    assert torch.allclose(again, moved, atol=1e-4)
