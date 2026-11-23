"""Molt web app + HTTP API (FastAPI). Run:  python -m molt.server --model path/to/model.pth
Then open http://127.0.0.1:3000 .

Streaming endpoints use Server-Sent Events: one `data: {json}` line per event.
One GPU, one robot: model calls are serialised with a lock.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .probes import ProbeSet
from .robot import LearnConfig, Robot, chat_segments, clean
from .sampling import SamplingConfig

WEB = Path(__file__).parent / "web"


def sse(events):
    for e in events:
        yield "data: " + json.dumps(e, ensure_ascii=False) + "\n\n"


def build_app(robot: Robot, probes: ProbeSet | None = None) -> FastAPI:
    app = FastAPI(title="Molt")
    lock = threading.RLock()
    probes = probes or ProbeSet()
    app.state.robot, app.state.probes = robot, probes
    curve: list[dict] = []  # every trained chunk, so a page reload keeps the learning curve

    def info():
        c = robot.model.cfg
        return {"variant": c.variant, "n_layer": c.n_layer, "n_embd": c.n_embd, "n_head": c.n_head,
                "lora_r": robot.lora_r, "tune_ego": robot.ego_wkv.requires_grad, "anchors": len(robot.anchors),
                "sampling": asdict(robot.sampling), "learn": asdict(robot.learn_cfg), "history": robot.history[-50:],
                "probes": [asdict(p) for p in probes.probes], "rounds": probes.rounds, "curve": curve}

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/info")
    def get_info():
        return info()

    @app.get("/api/memory")
    def memory():
        with lock:
            return robot.memory_stats()

    @app.post("/api/config")
    def config(body: dict):
        with lock:
            for k, v in (body.get("sampling") or {}).items():
                if hasattr(robot.sampling, k) and k != "stop":
                    setattr(robot.sampling, k, type(getattr(robot.sampling, k))(v))
            robot.set_learn_config(**(body.get("learn") or {}))
        return info()

    @app.post("/api/chat")
    def chat(body: dict):
        prompt = f"User: {clean(body.get('prompt', ''))}\n\nAssistant:" + (" <think" if body.get("think") else "")

        def run():
            with lock:
                yield from robot.speak_iter(prompt)
                yield {"memory": robot.memory_stats()}
        return StreamingResponse(sse(run()), media_type="text/event-stream")

    @app.post("/api/learn")
    def learn(body: dict):
        if body.get("assistant"):
            data = [chat_segments([("user", body.get("user", "")), ("assistant", body["assistant"])])]
        elif body.get("text"):
            data = body["text"]
        else:
            raise HTTPException(400, "need text, or user + assistant")
        epochs = int(body.get("epochs", 1))
        label = body.get("label", "learn")
        q: queue.Queue = queue.Queue()

        def on_chunk(d):
            curve.append({**d, "label": label})
            q.put({"chunk": d})

        def work():
            with lock:
                try:
                    rec = robot.learn(data, epochs=epochs, on_chunk=on_chunk)
                    q.put({"done": rec})
                    if probes.probes:
                        q.put({"round": probes.evaluate(robot, label=label)})
                    q.put({"memory": robot.memory_stats()})
                except Exception as e:  # surface errors to the page
                    q.put({"error": repr(e)})
                finally:
                    q.put(None)
        threading.Thread(target=work, daemon=True).start()
        return StreamingResponse(sse(iter(q.get, None)), media_type="text/event-stream")

    @app.post("/api/tokens")
    def tokens(body: dict):
        with lock:
            st = robot.ego if body.get("from", "ego") == "ego" else robot.state
            return robot.token_losses(body.get("text", ""), state=st, prefix=body.get("prefix", ""))

    @app.post("/api/reset")
    def reset(body: dict):
        with lock:
            robot.reset(body.get("mode", "ego"), float(body.get("ratio", 0.5)))
            return robot.memory_stats()

    @app.post("/api/probes")
    def probe_ops(body: dict):
        with lock:
            act = body.get("action")
            if act == "add":
                probes.add(body.get("prompt", ""), body.get("expect", ""))
            elif act == "remove":
                probes.remove(int(body["id"]))
            elif act == "eval":
                probes.evaluate(robot, label=body.get("label", "manual"))
            return {"probes": [asdict(p) for p in probes.probes], "rounds": probes.rounds}

    @app.post("/api/anchors")
    def anchors(body: dict):
        with lock:
            robot.anchors = []
            robot.make_anchors([p for p in body.get("prompts", []) if p.strip()])
            return {"anchors": len(robot.anchors)}

    @app.post("/api/save")
    def save(body: dict):
        with lock:
            robot.save(body.get("path", "molt_memory"))
        return {"ok": True}

    @app.post("/api/load")
    def load(body: dict):
        with lock:
            robot.load(body.get("path", "molt_memory"))
            return robot.memory_stats()

    # ---- labs ----------------------------------------------------------------
    from .labs import LABS, optimize
    labs = {L.id: L(robot) for L in LABS}

    def lab_of(lid):
        if lid not in labs:
            raise HTTPException(404, f"no lab {lid}")
        return labs[lid]

    def background(job):
        """Run job(emit) in a worker thread under the model lock; stream its events as SSE."""
        q: queue.Queue = queue.Queue()

        def work():
            with lock:
                try:
                    q.put({"result": job(q.put)})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    q.put({"error": repr(e)})
                finally:
                    q.put({"interventions": robot.model.interventions()})
                    q.put(None)
        threading.Thread(target=work, daemon=True).start()
        return StreamingResponse(sse(iter(q.get, None)), media_type="text/event-stream")

    @app.get("/api/labs")
    def list_labs():
        return [lab.schema() for lab in labs.values()]

    @app.post("/api/labs/{lid}/run")
    def lab_run(lid: str, body: dict):
        lab = lab_of(lid)
        return background(lambda emit: lab.run(lab.values(body.get("values")), emit))

    @app.post("/api/labs/{lid}/optimize")
    def lab_optimize(lid: str, body: dict):
        lab = lab_of(lid)
        search = body.get("search") or [p.name for p in lab.params if p.search]
        return background(lambda emit: optimize(lab, lab.values(body.get("values")), search,
                                                int(body.get("trials", 8)), emit))

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

    @app.get("/api/interventions")
    def get_interventions():
        return robot.model.interventions()

    @app.post("/api/interventions/reset")
    def reset_interventions():
        with lock:
            robot.model.reset_interventions()
            from .lora import set_dial
            set_dial(robot.model, 1.0)
            return robot.model.interventions()

    @app.post("/api/dial")
    def dial(body: dict):
        from .lora import set_dial
        with lock:
            set_dial(robot.model, float(body.get("value", 1.0)))
        return {"ok": True}

    @app.post("/api/memory/restore")
    def restore_memory():
        with lock:
            robot.restore("fresh")
            return {"ok": True}

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anchor", type=float, default=3.0, help="KL weight to the frozen self (0 = off)")
    ap.add_argument("--tune-ego", action="store_true", help="also train the ego state (state tuning); off by default: "
                    "in our single-fact probes, writing to ego alone overwrote unrelated answers")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3000)
    a = ap.parse_args()
    import uvicorn
    robot = Robot(a.model, lora_r=a.lora_r, tune_ego=a.tune_ego, learn_cfg=LearnConfig(lr=a.lr, anchor_weight=a.anchor))
    if a.anchor > 0:
        print("building self-anchors ...", flush=True)
        robot.make_anchors(["What is the capital of Japan?", "Give me a tip for learning Python.",
                            "用一句话解释什么是神经网络。", "Write a haiku about rain.", "12 乘以 7 等于多少？",
                            "Who wrote Hamlet?"])
    print(f"Molt is running: http://{a.host}:{a.port}", flush=True)
    uvicorn.run(build_app(robot), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
