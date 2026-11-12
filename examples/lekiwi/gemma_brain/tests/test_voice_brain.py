"""voice_brain.py without a microphone: the fixed answers, the plan read back, an utterance cut
from the microphone's blocks by loudness."""

import queue

import numpy as np
import pytest
import voice_brain as vb
from common.planner import parse_plan


@pytest.mark.parametrize(
    ("heard", "options", "want"),
    [
        ("Goal.", ["go", "cancel"], "go"),  # whisper's "Go." on the owner's recording, Nov 11
        ("Go!", ["go", "cancel"], "go"),
        ("Start!", ["go", "cancel"], "go"),
        ("Cancel.", ["go", "cancel"], "cancel"),
        ("开始。", ["go", "cancel"], "go"),
        ("取消", ["go", "cancel"], "cancel"),
        ("Pick the tennis ball and put it in the box.", ["go", "cancel"], None),  # a sentence
        ("Go watch tonight.", ["go", "cancel"], None),  # Gemma E2B's "Go."
        ("Hello", ["go", "cancel"], None),
        ("", ["go", "cancel"], None),
        ("Stop!", ["stop"], "stop"),
        ("停。", ["stop"], "stop"),
        ("Going.", ["stop"], None),  # the robot's own "Going." heard back
        ("Got the red brick.", ["stop"], None),
    ],
)
def test_a_short_answer_is_the_nearest_allowed_word(heard, options, want):
    assert vb.nearest(heard, options) == want


def test_the_plan_is_read_back_as_a_sentence():
    steps = parse_plan(
        'pick "tennis ball"\nplace "tennis ball" in "box"\npick "red brick"\nplace "red brick" in "bowl"\ndone'
    )
    assert (
        vb.plan_words(steps)
        == "pick the tennis ball, put it in the box, then pick the red brick, put it in the bowl"
    )


def _mic(blocks):
    m = vb.Mic.__new__(vb.Mic)
    m.blocks, m.noise = queue.Queue(), 50.0
    m.drain = lambda: None  # the blocks below are what is "heard" from now on
    for b in blocks:
        m.blocks.put(b)
    return m


def test_an_utterance_runs_from_the_first_loud_block_to_a_second_of_quiet():
    n = int(vb.RATE * vb.BLOCK_S)
    quiet, loud = np.full(n, 20, np.int16), np.full(n, 3000, np.int16)
    gap = int(vb.QUIET_S / vb.BLOCK_S) + 1
    m = _mic([quiet] * 10 + [loud] * 20 + [quiet] * 5 + [loud] * 5 + [quiet] * gap + [loud] * 3)
    got = m.utterance()
    # both bursts (the short pause inside it), then the quiet that ended it; not the later burst
    assert len(got) == n * (20 + 5 + 5 + gap)


def test_nothing_said_in_time_is_none():
    n = int(vb.RATE * vb.BLOCK_S)
    m = _mic([np.full(n, 20, np.int16)] * 10)
    assert m.utterance(wait_s=0.0) is None
