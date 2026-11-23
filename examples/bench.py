"""Speed and memory on your GPU.  python examples/bench.py path/to/model.pth"""
import sys
import time

import torch

from molt import RWKV7
from molt.decode import GraphDecoder
from molt.robot import LearnConfig, Robot


def timed(fn, n=3):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n


path = sys.argv[1]
m = RWKV7.from_pretrained(path)
c = m.cfg
print(f"{path}\nvariant {c.variant}, L{c.n_layer} D{c.n_embd}, {sum(p.numel() for p in m.parameters())/1e9:.2f}B params")
print(f"GPU {torch.cuda.get_device_name()}")
for T in (1024, 8192):
    idx = torch.randint(0, c.vocab_size, (1, T), device="cuda")
    with torch.no_grad():
        dt = timed(lambda: m(idx, last_only=True))
    print(f"prefill  T={T:5d}: {T / dt:9,.0f} tok/s")

x = torch.tensor([[11]], device="cuda")
with torch.no_grad():
    st = m.new_state()
    dt = timed(lambda: m(x, st, last_only=True), 20)
print(f"decode   eager     : {1 / dt:9,.1f} tok/s")
if not m.has_attention:
    dec = GraphDecoder(m)
    dt = timed(lambda: dec.step(11), 100)
    print(f"decode   CUDA graph: {1 / dt:9,.1f} tok/s")
del m
torch.cuda.empty_cache()

bot = Robot(path, lora_r=16, learn_cfg=LearnConfig(chunk_len=1024, bptt=2))
text = "RWKV is an RNN with transformer-level performance. " * 400
bot.learn(text[:2000])  # warm-up: Triton autotuning
torch.cuda.reset_peak_memory_stats()
t = time.time()
r = bot.learn(text)
torch.cuda.synchronize()
print(f"learn (LoRA r16, chunk 1024, bptt 2): {r['tokens'] / (time.time() - t):,.0f} tok/s, "
      f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
