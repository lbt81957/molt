"""Telemetry bus: what the model perceives, remembers and learns, as a stream of small events.

Producers (Robot, labs) call bus.publish(kind, **data); consumers (the web page's pulse dock)
subscribe and get every event plus a short replay of recent history. Events:

  activity  {"state": idle | listening | speaking | learning | lab, "label"}
  tok       one generated token: {"s": surprise in nats, "d": [per-layer |dWKV|], "t": text}
  listen    one prefill chunk:   {"n": tokens, "d": [per-layer |dWKV|], "s": mean surprise or None}
  step      one optimizer step:  {"loss", "kl", "gnorm", "lora": [per-layer |dW|]}
  progress  {"job", "phase", "done", "total"}
"""
from __future__ import annotations

import collections
import queue
import threading
import time
from contextlib import contextmanager


class Bus:
    def __init__(self, history: int = 1500):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.recent = collections.deque(maxlen=history)
        self.activity = "idle"

    def publish(self, kind: str, **data):
        ev = {"k": kind, "ts": time.time(), **data}
        with self._lock:
            self.recent.append(ev)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=5000)
        with self._lock:
            for ev in self.recent:
                q.put_nowait(ev)
            self._subs.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    @contextmanager
    def doing(self, state: str, label: str = ""):
        """Mark an activity for its duration (nested activities restore the outer one)."""
        prev = self.activity
        self.activity = state
        self.publish("activity", state=state, label=label)
        try:
            yield
        finally:
            self.activity = prev
            self.publish("activity", state=prev, label="")


class NullBus(Bus):
    """Used when nobody listens: publishing costs nothing."""

    def publish(self, kind, **data):
        pass
