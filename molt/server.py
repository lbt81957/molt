"""HTTP API around one Robot (FastAPI). Run:  python -m molt.server --model path/to/model.pth

Endpoints mirror the Robot methods. /speak streams plain text when "stream": true.
One GPU, one robot: requests are serialised with a lock.
"""
from __future__ import annotations

import argparse
import queue
import threading

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .robot import LearnConfig, Robot, chat_segments
from .sampling import SamplingConfig


class SpeakReq(BaseModel):
    prompt: str = ""
    chat: bool = False          # wrap prompt as "User: ...\n\nAssistant:"
    think: bool = False
    stream: bool = False
    temperature: float = 1.0
    top_p: float = 0.3
    max_tokens: int = 512


class TextReq(BaseModel):
    text: str


class LearnReq(BaseModel):
    texts: list[str] = []
    dialogue: list[tuple[str, str]] = []  # [("user", ...), ("assistant", ...)]
    epochs: int = 1
    absorb: bool = False


class ResetReq(BaseModel):
    mode: str = "ego"
    ratio: float = 0.5


class PathReq(BaseModel):
    path: str


def build_app(robot: Robot) -> FastAPI:
    app = FastAPI(title="Molt")
    lock = threading.Lock()

    @app.post("/listen")
    def listen(r: TextReq):
        with lock:
            robot.listen(r.text)
        return {"ok": True}

    @app.post("/speak")
    def speak(r: SpeakReq):
        cfg = SamplingConfig(temperature=r.temperature, top_p=r.top_p, max_tokens=r.max_tokens)
        prompt = r.prompt
        if r.chat:
            prompt = f"User: {r.prompt.strip()}\n\nAssistant:" + (" <think" if r.think else "")
        if not r.stream:
            with lock:
                return {"text": robot.speak(prompt, cfg=cfg)}
        q: queue.Queue = queue.Queue()

        def work():
            with lock:
                try:
                    robot.speak(prompt, cfg=cfg, stream=q.put)
                finally:
                    q.put(None)
        threading.Thread(target=work, daemon=True).start()
        return StreamingResponse(iter(q.get, None), media_type="text/plain; charset=utf-8")

    @app.post("/learn")
    def learn(r: LearnReq):
        data = list(r.texts)
        if r.dialogue:
            data.append(chat_segments(r.dialogue))
        with lock:
            return robot.learn(data, epochs=r.epochs, absorb=r.absorb)

    @app.post("/reflect")
    def reflect(r: TextReq):
        with lock:
            return robot.reflect(r.text)

    @app.post("/reset")
    def reset(r: ResetReq):
        with lock:
            robot.reset(r.mode, r.ratio)
        return {"ok": True}

    @app.post("/save")
    def save(r: PathReq):
        with lock:
            robot.save(r.path)
        return {"ok": True}

    @app.post("/load")
    def load(r: PathReq):
        with lock:
            robot.load(r.path)
        return {"ok": True}

    @app.get("/info")
    def info():
        c = robot.model.cfg
        return {"variant": c.variant, "n_layer": c.n_layer, "n_embd": c.n_embd, "lora_r": robot.lora_r,
                "history": robot.history[-20:]}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3000)
    a = ap.parse_args()
    import uvicorn
    robot = Robot(a.model, lora_r=a.lora_r, learn_cfg=LearnConfig(lr=a.lr))
    uvicorn.run(build_app(robot), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
