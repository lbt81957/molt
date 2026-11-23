"""Which memory tier should a new fact go to?  python examples/memory_tiers.py path/to/model.pth

Teaches one fact by repeated online updates, then asks the trained question, a paraphrase and two
unrelated questions. Compares writing into LoRA adapters, into the ego state, and LoRA + self-anchor KL.
"""
import sys, torch
from molt.robot import Robot, chat_segments, SamplingConfig, LearnConfig
P = sys.argv[1]
cfg = SamplingConfig(temperature=0, max_tokens=50)
q = "What is the name of the snake framework that learns while chatting?"
q2 = "Which framework lets an RWKV model learn while you chat with it?"   # paraphrase, never trained
probe = [q, q2, "What is the capital of France?", "Write one sentence about cats."]
anchors = ["What is the capital of Japan?", "Give me a tip for learning Python.", "Write a haiku about rain.",
           "What is 12 times 7?", "Explain what a neural network is in one sentence.", "Who wrote Hamlet?"]
fact = [("user", q), ("assistant", "It is called Molt. Molt lets an RWKV model learn online while you chat with it.")]
for name, kw, lc in [("lora", dict(lora_r=16, tune_ego=False), LearnConfig(lr=3e-4)),
                     ("ego", dict(lora_r=0, tune_ego=True), LearnConfig(ego_lr=1e-2)),
                     ("lora+anchor", dict(lora_r=16, tune_ego=False), LearnConfig(lr=3e-4, anchor_weight=3.0))]:
    bot = Robot(P, learn_cfg=lc, **kw)
    if "anchor" in name: bot.make_anchors(anchors)
    for i in range(8): r = bot.learn([chat_segments(fact)])
    print(f"== {name}: final train loss {r['loss_last']:.3f}")
    for p in probe:
        bot.reset(); print("  ", p[:30], "->", repr(bot.chat(p, cfg=cfg))[:100])
    del bot; torch.cuda.empty_cache()
