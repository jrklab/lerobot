#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Goal-level task plans from Gemma 4 26B: a task sentence and the two camera frames in,
2-8 lines of `pick "x"` / `place "x" <rel> "y"` / `push "x" toward "y"` out, limited by a
grammar. The planner only names verbs, objects and relations; code expands each verb into
skills. Benchmarked Oct 26: 15 of 16 reworded tasks correct in ~0.8 s.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

RELATIONS = ("in", "on", "next_to", "left_of", "right_of", "front_of", "behind")

STEPS_HELP = """Steps:
pick <obj>
place <obj> <rel> <obj>      rel: in on next_to left_of right_of front_of behind
push <obj> toward <obj>
<obj> is a short quoted noun phrase naming one thing with its colour: the task's colour when it
gives one, else the colour you see in the images (the robot finds objects by these names)."""

PLAN_SYSTEM = f"""You control a mobile robot arm. Image 1: front camera. Image 2: wrist camera.
Turn the task into steps, one per line, then done. Start from what the gripper holds now:
an object already held needs no pick. {STEPS_HELP}
Example. Task: put the yellow block on top of the blue block
pick "yellow block"
place "yellow block" on "blue block"
done
Example. Task: put the yellow block in the bowl (the images show an orange bowl)
pick "yellow block"
place "yellow block" in "orange bowl"
done"""

REPLAN_SYSTEM = f"""You control a mobile robot arm. Image 1: front camera. Image 2: wrist camera.
A step of the task failed. Write the steps that still need doing, one per line, then done.
Start from what the robot holds now. {STEPS_HELP}"""

PLAN_GBNF = r'''root ::= (step "\n"){1,8} "done"
step ::= "pick " o | "place " o " " rel " " o | "push " o " toward " o
rel ::= "in" | "on" | "next_to" | "left_of" | "right_of" | "front_of" | "behind"
o ::= "\"" [a-z ]{1,24} "\""'''

_STEP = re.compile(r'^(pick|place|push)\s+"([^"]+)"(?:\s+(\w+)\s+"([^"]+)")?$')


@dataclass(frozen=True)
class Step:
    verb: str
    obj: str
    rel: str | None = None
    target: str | None = None

    def __str__(self) -> str:
        if self.verb == "pick":
            return f'pick "{self.obj}"'
        return f'{self.verb} "{self.obj}" {self.rel} "{self.target}"'


def parse_plan(text: str) -> list[Step]:
    """The grammar's lines -> Steps. Raises ValueError on anything else."""
    steps = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line == "done":
            continue
        m = _STEP.match(line)
        if not m:
            raise ValueError(f"not a plan step: {line!r}")
        verb, obj, rel, target = m.groups()
        if verb == "pick" and rel:
            raise ValueError(f"pick takes one object: {line!r}")
        if verb == "place" and rel not in RELATIONS:
            raise ValueError(f"unknown relation: {line!r}")
        if verb == "push" and rel != "toward":
            raise ValueError(f"push needs 'toward': {line!r}")
        steps.append(Step(verb, obj.strip(), rel, target.strip() if target else None))
    if not steps:
        raise ValueError("empty plan")
    return steps


def objects_in(steps: list[Step]) -> list[str]:
    """Every object the plan names, in order of first mention."""
    seen = []
    for s in steps:
        for o in (s.obj, s.target):
            if o and o not in seen:
                seen.append(o)
    return seen


def check_plan(steps: list[Step], supported: set[tuple[str, str | None]], holding: str | None) -> list[str]:
    """Problems that stop the plan before anything moves: a verb/relation the skills can't do,
    or a pick/place order that can't work with one gripper. Empty = runnable."""
    problems = []
    held = holding
    for i, s in enumerate(steps, 1):
        if (s.verb, s.rel) not in supported:
            problems.append(f"step {i} ({s}): the robot can't do '{s.verb}{' ' + s.rel if s.rel else ''}' yet")
        if s.verb == "pick":
            if held:
                problems.append(f"step {i} ({s}): already holding the {held}")
            held = s.obj
        elif s.verb == "place":
            if held != s.obj:
                problems.append(f"step {i} ({s}): not holding the {s.obj}")
            held = None
        elif s.verb == "push" and held:
            problems.append(f"step {i} ({s}): can't push while holding the {held}")
    return problems


class Planner:
    def __init__(self, model):
        """model: gemma.LlamaCpp pointed at the 26B server."""
        self.model = model

    def plan(self, task: str, front: bytes, wrist: bytes, holding: str | None = None):
        text = f"Task: {task}\nThe gripper now holds: {holding or 'nothing'}"
        a = self.model.generate(text, [front, wrist], system=PLAN_SYSTEM, grammar=PLAN_GBNF, max_tokens=120)
        return parse_plan(a.text), a

    def replan(self, task: str, done: list[Step], failed: Step, reason: str, holding: str | None,
               front: bytes, wrist: bytes):
        lines = [f"Task: {task}"]
        lines.append("Done so far: " + ("; ".join(map(str, done)) if done else "nothing"))
        lines.append(f"Failed: {failed} ({reason})")
        lines.append(f"The gripper now holds: {holding or 'nothing'}")
        a = self.model.generate("\n".join(lines), [front, wrist], system=REPLAN_SYSTEM, grammar=PLAN_GBNF,
                                max_tokens=120)
        return parse_plan(a.text), a
