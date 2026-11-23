"""RWKV World tokenizer (rwkv_vocab_v20230424): greedy longest-match over bytes.

Same algorithm as the reference tokenizer in BlinkDL/RWKV-LM, re-implemented with
ast.literal_eval instead of eval and a decode that tolerates split UTF-8 sequences.
"""
from __future__ import annotations

import ast
from pathlib import Path

DEFAULT_VOCAB = Path(__file__).parent / "assets" / "rwkv_vocab_v20230424.txt"


class RWKVTokenizer:
    def __init__(self, vocab_file: str | Path = DEFAULT_VOCAB):
        self.idx2bytes: dict[int, bytes] = {}
        for line in Path(vocab_file).read_text(encoding="utf-8").splitlines():
            idx = int(line[: line.index(" ")])
            tok = ast.literal_eval(line[line.index(" ") : line.rindex(" ")].strip())
            tok = tok.encode("utf-8") if isinstance(tok, str) else tok
            assert len(tok) == int(line[line.rindex(" ") :]), line
            self.idx2bytes[idx] = tok
        self.bytes2idx = {b: i for i, b in self.idx2bytes.items()}
        # candidates[b0][b1]: tokens starting with bytes b0 b1, longest first
        self.candidates: list[list[list[bytes]]] = [[[] for _ in range(256)] for _ in range(256)]
        self.maxlen = [1] * 256
        for tok in sorted(self.bytes2idx, key=len, reverse=True):
            if len(tok) >= 2:
                self.candidates[tok[0]][tok[1]].append(tok)
                self.maxlen[tok[0]] = max(self.maxlen[tok[0]], len(tok))

    @property
    def vocab_size(self) -> int:
        return 65536

    def encode_bytes(self, src: bytes) -> list[int]:
        out, i, n = [], 0, len(src)
        while i < n:
            tok = src[i : i + 1]
            if i + 1 < n:
                window = src[i : i + self.maxlen[src[i]]]
                for cand in self.candidates[src[i]][src[i + 1]]:
                    if window.startswith(cand):
                        tok = cand
                        break
            out.append(self.bytes2idx[tok])
            i += len(tok)
        return out

    def encode(self, text: str) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def decode_bytes(self, tokens) -> bytes:
        return b"".join(self.idx2bytes.get(int(t), b"") for t in tokens)

    def decode(self, tokens, errors: str = "replace") -> str:
        return self.decode_bytes(tokens).decode("utf-8", errors=errors)
