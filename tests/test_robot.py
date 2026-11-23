"""Robot behaviour on a real checkpoint (skipped without CUDA or the model file)."""
import os
from pathlib import Path

import pytest
import torch

MODEL = Path(os.environ.get("MOLT_TEST_MODELS", "D:/models/rwkv")) / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and MODEL.exists()), reason="needs CUDA + 0.1B model")


@pytest.fixture(scope="module")
def bot():
    from molt.robot import Robot
    return Robot(MODEL, lora_r=8)


@pytest.mark.parametrize("prompt", ["User: 用中文介绍一下你自己。\n\nAssistant:", "User: Say hi.\n\nAssistant:"])
def test_streamed_tokens_carry_their_own_text(bot, prompt):
    from molt.sampling import SamplingConfig
    bot.reset()
    ev = list(bot.speak_iter(prompt, SamplingConfig(temperature=0, max_tokens=60)))
    reply, toks = ev[-1]["text"], [e for e in ev if "id" in e]
    assert "".join(e["text"] for e in toks) == reply
    for e in toks:  # whole-character tokens: the attached text is exactly the token
        own = bot.tok.decode([e["id"]])
        if e["text"] and "\ufffd" not in own:
            assert e["text"] == own


def test_learn_then_recall_and_reset(bot):
    from molt.robot import chat_segments
    from molt.sampling import SamplingConfig
    q = "What is the secret word?"
    before = bot.token_losses(" The secret word is ouroboros.", state=bot.ego, prefix=f"User: {q}\n\nAssistant:")
    for _ in range(6):
        bot.learn([chat_segments([("user", q), ("assistant", "The secret word is ouroboros.")])])
    after = bot.token_losses(" The secret word is ouroboros.", state=bot.ego, prefix=f"User: {q}\n\nAssistant:")
    mean = lambda t: sum(x["loss"] for x in t) / len(t)
    assert mean(after) < mean(before) * 0.5
    bot.reset()
    assert "ouroboros" in bot.chat(q, cfg=SamplingConfig(temperature=0, max_tokens=30)).lower()
