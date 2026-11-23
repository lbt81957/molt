"""Fast token-by-token decoding with CUDA graphs.

Decoding one token launches ~50 small kernels per layer. On Windows (WDDM) launch overhead
dominates and eager decoding of a 0.1B model crawls at ~20 tok/s. Capturing the whole
one-token forward (state update included) into a CUDA graph removes that overhead.
"""
from __future__ import annotations

import torch

from .state import State


class GraphDecoder:
    def __init__(self, model, batch: int = 1):
        assert next(model.parameters()).is_cuda, "GraphDecoder needs a CUDA model"
        self.model, self.batch = model, batch
        self.tok = torch.zeros(batch, 1, dtype=torch.long, device="cuda")
        self.state = model.new_state(batch)
        self.graph = None

    def _capture(self):
        m = self.model

        def body():
            logits, out = m(self.tok, self.state, last_only=True)
            self.state.att_x.copy_(out.att_x)
            self.state.wkv.copy_(out.wkv)
            self.state.ffn_x.copy_(out.ffn_x)
            return logits[:, -1]

        saved = self.state.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.logits = body()
        self.set_state(saved)

    # ---- state I/O (copies, the graph owns its buffers) -------------------
    def set_state(self, state: State | None):
        if state is None:
            for t in (self.state.att_x, self.state.wkv, self.state.ffn_x):
                t.zero_()
        else:
            self.state.att_x.copy_(state.att_x)
            self.state.wkv.copy_(state.wkv)
            self.state.ffn_x.copy_(state.ffn_x)

    def get_state(self) -> State:
        return self.state.clone()

    @torch.no_grad()
    def step(self, token) -> torch.Tensor:
        """Feed one token (int or [B] tensor); advance the internal state; return logits [B, V]."""
        if self.graph is None:
            self._capture()
        if isinstance(token, int):
            self.tok.fill_(token)
        else:
            self.tok.copy_(torch.as_tensor(token, device="cuda").view(self.batch, 1))
        self.graph.replay()
        return self.logits
