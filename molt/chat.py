"""Terminal chat with online learning.  python -m molt.chat --model path/to/model.pth

Commands:
  /fix <better reply>   learn: replace the last reply with yours (human-in-the-loop correction)
  /learn <text>         learn a piece of text
  /reset [ego|zero]     clear working memory
  /think                toggle think mode
  /save <dir> | /load <dir>
  /quit
"""
from __future__ import annotations

import argparse
import sys

from .robot import LearnConfig, Robot, chat_segments
from .sampling import SamplingConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anchor", type=float, default=1.0, help="KL weight to the frozen self (0 = off)")
    a = ap.parse_args()
    bot = Robot(a.model, lora_r=a.lora_r, learn_cfg=LearnConfig(lr=a.lr, anchor_weight=a.anchor))
    if a.anchor > 0:
        print("building self-anchors ...", flush=True)
        bot.make_anchors(["What is the capital of Japan?", "Give me a tip for learning Python.",
                          "用一句话解释什么是神经网络。", "Write a haiku about rain.", "12 乘以 7 等于多少？"])
    cfg, think, last = SamplingConfig(), False, None
    print(__doc__)
    while True:
        try:
            line = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        if cmd == "/quit":
            break
        if cmd == "/fix" and last:
            print(bot.learn([chat_segments([("user", last[0]), ("assistant", arg)])], epochs=3))
            continue
        if cmd == "/learn":
            print(bot.learn(arg, epochs=3)); continue
        if cmd == "/reset":
            bot.reset(arg or "ego"); print("(memory cleared)"); continue
        if cmd == "/think":
            think = not think; print(f"(think mode {'on' if think else 'off'})"); continue
        if cmd in ("/save", "/load"):
            getattr(bot, cmd[1:])(arg or "molt_memory"); print("(ok)"); continue
        print("\nAssistant:", end="", flush=True)
        reply = bot.chat(line, think=think, cfg=cfg, stream=lambda s: print(s, end="", flush=True))
        last = (line, reply)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
