"""Molt web app + HTTP API (FastAPI). Run:  python -m molt.server --model path/to/model.pth
Then open http://127.0.0.1:3000 .

Streaming endpoints use Server-Sent Events: one `data: {json}` line per event.
One GPU, one robot: model calls are serialised with a lock. The model can be swapped at runtime.
"""
from __future__ import annotations

import argparse
import gc
import json
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .robot import LearnConfig, Robot, chat_segments, clean
from .telemetry import Bus

WEB = Path(__file__).parent / "web"
ANCHOR_PROMPTS = ["What is the capital of Japan?", "Give me a tip for learning Python.", "用一句话解释什么是神经网络。",
                  "Write a haiku about rain.", "12 乘以 7 等于多少？", "Who wrote Hamlet?"]


def sse(events):
    for e in events:
        yield "data: " + json.dumps(e, ensure_ascii=False) + "\n\n"


class Session:
    """The loaded model and everything built on it; replaced as a whole when the model is switched."""

    def __init__(self, bus: Bus, lora_r=16, lr=3e-4, anchor=3.0, tune_ego=False):
        self.bus, self.opts = bus, dict(lora_r=lora_r, lr=lr, anchor=anchor, tune_ego=tune_ego)
        self.robot: Robot | None = None
        self.labs: dict = {}
        self.path: str | None = None

    def load(self, path: str, emit=lambda e: None):
        from .labs import LABS
        o = self.opts
        if self.robot is not None:
            emit({"log": "释放当前模型…"})
            self.robot, self.labs = None, {}
            gc.collect()
            torch.cuda.empty_cache()
        emit({"log": f"加载 {Path(path).name} …"})
        self.bus.publish("progress", job="model", phase="加载权重", done=0, total=2)
        r = Robot(path, lora_r=o["lora_r"], tune_ego=o["tune_ego"], learn_cfg=LearnConfig(lr=o["lr"], anchor_weight=o["anchor"]))
        r.bus = self.bus
        if o["anchor"] > 0:
            emit({"log": "生成自我锚点…"})
            self.bus.publish("progress", job="model", phase="生成自我锚点", done=1, total=2)
            r.make_anchors(ANCHOR_PROMPTS)
        self.robot, self.path = r, str(Path(path).resolve())
        self.labs = {L.id: L(r) for L in LABS}
        self.bus.publish("progress", job="model", phase="就绪", done=2, total=2)
        self.bus.publish("model", path=self.path, info=self.info())
        emit({"log": "就绪"})
        return self.info()

    def info(self):
        r = self.robot
        c = r.model.cfg
        return {"path": self.path, "name": Path(self.path).name, "variant": c.variant, "n_layer": c.n_layer,
                "n_embd": c.n_embd, "n_head": c.n_head, "lora_r": r.lora_r, "tune_ego": r.ego_wkv.requires_grad,
                "anchors": len(r.anchors), "learn": asdict(r.learn_cfg), "history": r.history[-50:]}


def build_app(session: Session, models_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Molt")
    lock = threading.RLock()
    bus = session.bus
    rb = lambda: session.robot

    def background(job, label=""):
        """Run job(emit) in a worker thread under the model lock; stream its events as SSE."""
        q: queue.Queue = queue.Queue()

        def work():
            with lock, bus.doing("lab", label):
                try:
                    q.put({"result": job(q.put)})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    q.put({"error": repr(e)})
                finally:
                    if session.robot is not None:
                        q.put({"interventions": session.robot.model.interventions()})
                    q.put(None)
        threading.Thread(target=work, daemon=True).start()
        return StreamingResponse(sse(iter(q.get, None)), media_type="text/event-stream")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/info")
    def get_info():
        return session.info()

    # ---- pulse: everything the model perceives and learns, live ------------------
    @app.get("/api/pulse")
    def pulse():
        q = bus.subscribe()

        def stream():
            try:
                while True:
                    try:
                        yield q.get(timeout=2.0)
                    except queue.Empty:
                        free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
                        yield {"k": "hb", "ts": time.time(), "activity": bus.activity,
                               "vram_used": round((total - free) / 2**30, 2), "vram_total": round(total / 2**30, 2)}
            finally:
                bus.unsubscribe(q)
        return StreamingResponse(sse(stream()), media_type="text/event-stream")

    # ---- models ------------------------------------------------------------------
    def model_dirs():
        dirs = {Path(session.path).parent.resolve()} if session.path else set()
        if models_dir:
            dirs.add(Path(models_dir).resolve())
        return dirs

    @app.get("/api/models")
    def list_models():
        out = []
        for d in model_dirs():
            for f in sorted(d.glob("*.pth")):
                out.append({"path": str(f.resolve()), "name": f.name, "gb": round(f.stat().st_size / 2**30, 2)})
        return {"current": session.path, "models": out}

    @app.post("/api/models/load")
    def load_model(body: dict):
        path = body.get("path", "")
        if not Path(path).is_file():
            raise HTTPException(404, f"no such model: {path}")
        return background(lambda emit: session.load(path, emit), "切换模型")

    # ---- chat and learning ---------------------------------------------------------
    @app.post("/api/chat")
    def chat(body: dict):
        prompt = f"User: {clean(body.get('prompt', ''))}\n\nAssistant:" + (" <think" if body.get("think") else "")

        q: queue.Queue = queue.Queue()

        def work():  # generate in one worker thread: the lock must be released by the thread that took it
            with lock:
                try:
                    for ev in rb().speak_iter(prompt):
                        q.put(ev)
                except Exception as e:
                    q.put({"error": repr(e)})
                finally:
                    q.put(None)
        threading.Thread(target=work, daemon=True).start()
        return StreamingResponse(sse(iter(q.get, None)), media_type="text/event-stream")

    @app.post("/api/learn")
    def learn(body: dict):
        if body.get("assistant"):
            data = [chat_segments([("user", body.get("user", "")), ("assistant", body["assistant"])])]
        elif body.get("text"):
            data = body["text"]
        else:
            raise HTTPException(400, "need text, or user + assistant")
        epochs = int(body.get("epochs", 1))
        return background(lambda emit: {"done": rb().learn(data, epochs=epochs, on_chunk=lambda d: emit({"chunk": d}))},
                          "学习")

    @app.post("/api/reset")
    def reset(body: dict):
        with lock:
            rb().reset(body.get("mode", "ego"), float(body.get("ratio", 0.5)))
        return {"ok": True}

    @app.post("/api/dial")
    def dial(body: dict):
        from .lora import set_dial
        with lock:
            set_dial(rb().model, float(body.get("value", 1.0)))
        return {"ok": True}

    @app.post("/api/save")
    def save(body: dict):
        with lock:
            rb().save(body.get("path", "molt_memory"))
        return {"ok": True}

    @app.post("/api/load")
    def load(body: dict):
        with lock:
            rb().load(body.get("path", "molt_memory"))
        return {"ok": True}

    @app.post("/api/memory/restore")
    def restore_memory():
        with lock:
            rb().restore("fresh")
        return {"ok": True}

    # ---- interventions -------------------------------------------------------------
    @app.get("/api/interventions")
    def get_interventions():
        return rb().model.interventions()

    @app.post("/api/interventions/reset")
    def reset_interventions():
        from .lora import set_dial
        with lock:
            rb().model.reset_interventions()
            set_dial(rb().model, 1.0)
            return rb().model.interventions()

    # ---- labs ------------------------------------------------------------------------
    from .labs import optimize

    def lab_of(lid):
        if lid not in session.labs:
            raise HTTPException(404, f"no lab {lid}")
        return session.labs[lid]

    @app.get("/api/labs")
    def list_labs():
        return [lab.schema() for lab in session.labs.values()]

    @app.post("/api/labs/{lid}/run")
    def lab_run(lid: str, body: dict):
        lab = lab_of(lid)
        return background(lambda emit: lab.run(lab.values(body.get("values")), emit), lab.title)

    @app.post("/api/labs/{lid}/optimize")
    def lab_optimize(lid: str, body: dict):
        lab = lab_of(lid)
        search = body.get("search") or [p.name for p in lab.params if p.search]
        return background(lambda emit: optimize(lab, lab.values(body.get("values")), search, int(body.get("trials", 8)), emit),
                          lab.title + "（自动优化）")

    @app.post("/api/labs/{lid}/live")
    def lab_live(lid: str, body: dict):
        lab = lab_of(lid)
        with lock:
            return lab.live(lab.values(body.get("values"))) or {}

    @app.post("/api/labs/{lid}/act")
    def lab_act(lid: str, body: dict):
        lab = lab_of(lid)
        with lock:
            return lab.act(body.get("action", ""), {**(body.get("values") or {}), **(body.get("payload") or {})}) or {}

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--models-dir", default=None, help="extra folder to list models from (the --model folder is always listed)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anchor", type=float, default=3.0, help="KL weight to the frozen self (0 = off)")
    ap.add_argument("--tune-ego", action="store_true", help="also train the ego state (state tuning); off by default: "
                    "in our single-fact probes, writing to ego alone overwrote unrelated answers")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3000)
    a = ap.parse_args()
    import uvicorn
    session = Session(Bus(), lora_r=a.lora_r, lr=a.lr, anchor=a.anchor, tune_ego=a.tune_ego)
    session.load(a.model, emit=lambda e: print(e.get("log", ""), flush=True))
    print(f"Molt is running: http://{a.host}:{a.port}", flush=True)
    uvicorn.run(build_app(session, a.models_dir), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
