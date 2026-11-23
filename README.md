# Molt 🐍

**An RWKV model that learns while you use it.** Listen, speak, and learn in one loop, with
three memory tiers you can write to: the running state, a learned "ego" state, and LoRA weights.
Native on Windows: no MSVC, no WSL.

> A snake grows by shedding its skin and is still the same snake. Molt asks the same of a
> language model: keep updating from what it hears, and stay itself.

[中文说明](#中文说明)

Molt is a from-scratch successor to [RWKV-Ouroboros](https://github.com/neromous/RWKV-Ouroboros)
(2023), an online-training framework for RWKV with a human in the loop. Molt keeps its ideas and
its `listen / speak / learn` interface, and rebuilds everything else for the RWKV-7 generation.

## Highlights

- **Latest RWKV generations, checked against the official code.** RWKV-7 "Goose" (x070, the
  G1 series), plus the experimental x070a (deep-embedding FFN), x070b (DeepEmbedAttention hybrid) and
  RWKV-8 rc00 (x080). The test suite runs BlinkDL's own demo code as the reference: max logit
  difference is about 4e-5 in float32 for all four generations.
- **Windows-native fast kernels.** WKV-7 runs on [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
  Triton kernels through `triton-windows`, so nothing needs compiling. Gradients flow into the
  initial state, which is what makes state tuning and cross-chunk BPTT possible.
- **CUDA-graph decoding.** On Windows, eager decoding is limited by kernel-launch overhead.
  Molt captures the whole one-token step in a CUDA graph: 20 → 376 tok/s on a 0.1B model.
- **Three memory tiers.**
  - `state`: working memory, written by every `listen` and `speak`.
  - `ego`: a learned initial state (state tuning). `reset()` returns to it. Training it is opt-in
    (`tune_ego=True`): in our probes, writing facts into ego alone overwrote unrelated answers.
  - LoRA: float32 adapters over a frozen bf16 base, so tiny online updates are not rounded away.
- **Learning that carries context.** Long texts are split into chunks and the state is carried
  across chunk boundaries (truncated BPTT), so the model learns to use what it read earlier.
- **Self-anchoring against forgetting.** The model answers a few anchor prompts itself before
  learning. While it learns, a KL term keeps its outputs on those answers close to the frozen base.
  The snake eats its own tail.
- **Human-in-the-loop.** Includes an HTTP API (`molt.server`) and a terminal chat
  (`molt.chat`). In the chat, `/fix <better answer>` teaches the model your correction at once.

## The workbench

`python -m molt.server --model path.pth`, then open <http://127.0.0.1:3000>. It is a single page with no
build step, served by the same process.

![Molt workbench](docs/screenshot.png)

- **Chat with token-level surprise.** Each generated token is tinted by how unexpected it was to the
  model. Hover a token to see its probability and the model's top-5 alternatives.
- **Correct a reply and learn from it.** Rewrite any reply and the model learns your version at once.
  The loss curve and the anchor-KL curve update live, chunk by chunk.
- **Probes.** A fixed set of questions, re-run after every learning step. For each one you see the
  loss on the expected answer and its trend, plus whether the answer changed. Together they show what was
  acquired and what was damaged.
- **Memory maps.** Heatmaps of the working state (layer × head), how far it has drifted from ego,
  and the LoRA update norm per layer and module.
- **Microscope.** Paste any text to see per-token surprise. Set a baseline, learn something, then
  compare to see the per-token Δloss.

## Install (Windows, NVIDIA GPU)

```bat
conda create -n molt python=3.12 -y
conda activate molt
set PYTHONNOUSERSITE=1
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install "triton-windows>=3.7,<3.8" flash-linear-attention
pip install -e .[serve,dev]
```

`PYTHONNOUSERSITE=1` matters on Windows: packages in `%APPDATA%\Python` (an old torchvision, for
example) otherwise leak into the environment and break imports. Without fla, Molt falls back to
pure PyTorch, which is slow but still correct. Linux works too: use `triton` in place of
`triton-windows`.

Weights: <https://huggingface.co/BlinkDL/rwkv7-g1> (for example `rwkv7-g1l-1.5b-20261031-ctx34304.pth`).

## Quick start

```python
from molt.robot import Robot, LearnConfig, chat_segments

bot = Robot("rwkv7-g1l-1.5b-20261031-ctx34304.pth", lora_r=16,
            learn_cfg=LearnConfig(lr=3e-4, anchor_weight=3.0))
bot.make_anchors(["What is the capital of Japan?", "Write a haiku about rain."])

print(bot.chat("What is Molt?"))
bot.learn([chat_segments([("user", "What is Molt?"),
                          ("assistant", "Molt is a framework that lets RWKV learn while you chat with it.")])])
bot.reset()                       # clear working memory; what was learned stays in LoRA/ego
print(bot.chat("What is Molt?"))
print(bot.reflect("Molt lets RWKV learn online."))   # loss / perplexity under current memory
bot.save("my_molt")               # ego + adapters + working state (not the base weights)
```

Terminal chat: `python -m molt.chat --model path.pth`. HTTP server: `python -m molt.server --model path.pth`.

## A first look: which tier should a new fact go to?

`examples/memory_tiers.py` teaches one fact through 8 online updates. It then asks the trained
question, a paraphrase that was never trained, and two unrelated questions. Results on RWKV-7 G1l 1.5B
with greedy decoding (LoRA lr 3e-4, ego lr 1e-2, anchor weight 3, 6 anchors):

| written to | trained question | paraphrase | "What is the capital of France?" | "Write one sentence about cats." |
|---|---|---|---|---|
| LoRA only | ✅ | ✅ | ⚠️ "It is called Paris." (fact's template leaks) | ❌ "It is called Molt. …" |
| ego state only | ✅ | ✅ | ❌ repeats the fact | ❌ repeats the fact |
| LoRA + self-anchor | ✅ | ✅ | ✅ "The capital of France is Paris. …" | ✅ "Cats are small, furry animals …" |

G1d 0.1B shows the same pattern. With LoRA lr 1e-3, anchoring at weight 1 is enough on 0.1B but
only partly works on 1.5B. This is a single example, not an experiment. It shows the problem Molt
is built to study: in online learning, the hard part is not acquiring a fact but keeping everything
else intact.

## Speed (RTX 4060 Ti 16 GB, Windows 11, bf16)

| model | prefill 8k | decode (CUDA graph) | learn (LoRA r16) | peak VRAM when learning |
|---|---|---|---|---|
| G1d 0.1B | 61k tok/s | 376 tok/s | 2.2k tok/s | 1.3 GB |
| G1l 1.5B | 7.5k tok/s | 68 tok/s | 1.1k tok/s | 7.0 GB |

Reproduce with `python examples/bench.py path.pth`.

## Layout

| file | |
|---|---|
| `molt/model.py` | RWKV-7 as a trainable `nn.Module`; parameter names match the official checkpoints |
| `molt/variants.py` | x070a / x070b (DEA) / x080 |
| `molt/ops.py` | WKV-7 backends: `fla` (chunked Triton), `fla_rec`, `step` (decode), `torch` (reference) |
| `molt/state.py` | the recurrent state (float32 WKV), save / load / decay / mix |
| `molt/decode.py` | CUDA-graph decoder |
| `molt/robot.py` | `Robot`: listen / speak / learn / reflect / reset, memory tiers, anchors |
| `molt/lora.py` | float32 LoRA, merge back to an official-format `.pth` |
| `tests/` | parity against official code (all four generations), backend values and gradients, LoRA |

## What changed from RWKV-Ouroboros v1

v1 (2023–2024) was ahead of its time, and it also left some bugs and gaps that Molt fixes:

- In v1's RWKV-6 path, the WKV kernel never returned the new state. Inference therefore did not
  accumulate memory, and training restarted every chunk from the ego state. Molt carries the
  state everywhere and tests this against the official code.
- v1 kept the state in bf16 between steps. Molt keeps WKV in float32.
- v1's merge, decay and perplexity helpers had bugs. They are rewritten and tested.
- v1 had no evaluation of any kind. Molt starts from parity tests, benchmarks and a forgetting probe.

## Roadmap

Molt is the engineering base for **Ouroboros 2**, a study of multi-tier memory for online
continual learning on real-world chat streams. The question: which information should be written
to which tier, and does what was written actually get used later? Next steps:

- consolidation from state to weights (sleep-like replay);
- streaming benchmarks with forward-transfer and retention metrics;
- baselines (RAG, long context, TTT and Titans-style layers via fla).

## License and credits

Apache-2.0. Model math and the vendored test references (`tests/official/`) come from
[BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) (Apache-2.0); so does the RWKV World
vocabulary in `molt/assets/`. Fast kernels come from
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT; a dependency, not
vendored). The idea and the interface come from [RWKV-Ouroboros](https://github.com/neromous/RWKV-Ouroboros)
by neromous and contributors (including the author of Molt, who wrote its WebUI). No v1 code is
reused. See `NOTICE`.

---

## 中文说明

**Molt 是一个"边用边学"的 RWKV 框架**：听（listen）、说（speak）、学（learn）在同一个循环里完成。模型有三层可写的记忆：运行中的 state、学出来的 ego state、LoRA 权重。原生支持 Windows，不需要 MSVC，也不需要 WSL。

蛇靠蜕皮长大，蜕完皮还是那条蛇。Molt 对语言模型的要求也一样：不断从听到的内容中更新，同时不丢掉自己。

Molt 是 [RWKV-Ouroboros](https://github.com/neromous/RWKV-Ouroboros)（2023）的重写版。它保留了 v1 的人在回路思想和 `listen / speak / learn` 接口，其余部分全部面向 RWKV-7 重新实现。

- **代际支持**：RWKV-7 G1（x070），以及实验架构 x070a、x070b（DEA 混合注意力）、RWKV-8 rc00（x080）。测试直接执行 BlinkDL 官方 demo 代码作为参考，四代模型在 fp32 下的 logit 最大误差约 4e-5。
- **Windows 上的高速算子**：通过 `triton-windows` 使用 fla 的 Triton kernel，无需编译。梯度可以回传到初始 state，所以 state tuning 和跨段 BPTT 都能做。
- **CUDA Graph 解码**：Windows 上逐个启动 kernel 的开销很大，把单 token 前向整体捕获后，0.1B 模型的解码速度从 20 提到 376 tok/s，1.5B 为 69 tok/s。
- **三层记忆与自我锚定**：学习新知识时，用 KL 约束模型在自己生成的锚点回答上不偏离学习前的自己，从而抑制灾难性干扰。在 1.5B 上的单例演示中，只写 LoRA 或只写 ego state 时，模型回答无关问题也会复读新学的事实；加上自我锚定后，事实学会了，无关问题也照常回答（见上文表格）。
- **省显存**：训练时默认开启梯度检查点，交叉熵按块计算。1.5B 在线学习的显存峰值从 16.1 GB 降到 7.1 GB。

- **工作台前端**：`python -m molt.server --model 模型.pth` 后打开 http://127.0.0.1:3000 。
  - 对话按 token 着色，显示每个 token 的意外度，悬停可看候选词；
  - 任何回复都能改写后立即学习，学习曲线实时更新；
  - 考题每学一次自动重考，同时看到学会了什么、弄坏了什么；
  - 记忆热力图显示三层记忆的变化；
  - 显微镜对比学习前后每个 token 的 Δloss。

安装见上文 Install 一节。注意在 Windows 上设置 `PYTHONNOUSERSITE=1`。
