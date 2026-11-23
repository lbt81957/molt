# Development notes

## Environment (Windows)

- conda env `ouro`: Python 3.12, torch 2.11 cu130, triton-windows 3.7, flash-linear-attention 0.6.
- **Always set `PYTHONNOUSERSITE=1`**. Packages in `%APPDATA%\Python` otherwise leak in and break imports
  (for example an old torchvision).
- Test checkpoints live in `D:/models/rwkv` (override with `MOLT_TEST_MODELS`):
  - `rwkv7-g1d-0.1b-20260129-ctx8192.pth` (fast loop)
  - `rwkv7-g1l-1.5b-20261031-ctx34304.pth` (results)
  - `rwkv7a-g1d-0.1b`, `rwkv7b-g1b-0.1b`, `rwkv-8-pile-rc00` (parity tests)
- Run: `python -m molt.server --model D:/models/rwkv/<file>.pth --port 3000`
- Tests: `python -m pytest -q tests` (about 1 min; GPU tests skip without CUDA or checkpoints).
- Browser checks: Playwright with the installed Chrome (`p.chromium.launch(channel="chrome")`); no browser download needed.

## Map

| layer | files | notes |
|---|---|---|
| model | `model.py`, `variants.py`, `ops.py`, `state.py` | x070 / x070a / x070b / x080; state layout `[L,B,H,K,V]` (fla); WKV always float32 |
| interventions | `model.py` (`decay_scale`, `head_scale`, `steer`), `lora.py` (`dial`) | in-place tensors, so the CUDA graph follows them; defaults are a no-op |
| decoding | `decode.py` | CUDA graph; the eager path is used for x070b (its KV cache grows) |
| robot | `robot.py` | listen / speak_iter / learn / ask / token_losses / snapshot / restore / training() |
| strategies | `strategies.py` | sft / augment / distill / sleep; each reports progress on `robot.bus` |
| labs | `labs/` | `Lab.run / live / act / score`, `optimize()`; viz dicts rendered by `web/app.js` |
| telemetry | `telemetry.py` | `Bus.publish(kind, ...)`: tok / listen / step / progress / activity / model |
| server | `server.py` | `Session` holds robot + labs (hot-swappable); all model work runs in worker threads under one lock |
| web | `web/index.html`, `app.js`, `pulse.js`, `style.css` | no build step; `pulse.js` draws the dock on a canvas |

## Pitfalls already hit

- Never hold the model lock inside a generator served by Starlette. It advances generators on different
  threads, so the lock would be released by a thread that never took it. Do the work in a worker thread
  and stream through a queue (`background()` in `server.py`).
- The fla chunk kernel is wrong on random, out-of-distribution initial states. Gradients and values match
  the reference on states the model itself produced. Test with natural states.
- G1 models end a finished reply with token 0, so chat training data must too (`chat_segments`).
- Writing facts into the ego state alone (state tuning) overwrites unrelated answers. `tune_ego` is opt-in.
- Lab results from single runs are noisy: one compose question flips a column from 0% to 100%. Use bigger
  packs and several seeds before drawing conclusions.

## Next

1. Streaming lab: facts arrive over "days"; measure retention of earlier days (`sleep` vs online `sft`).
2. Larger fact packs with seeds; report mean ± spread.
3. DPO-style loss from rewritten replies (reference = LoRA disabled); KTO for thumbs up/down.
4. Teacher-written exams for anything learned (OpenAI-compatible endpoint, optional).
5. Batched decoding, then RL over the write policy (what to write, to which tier, for how long).
