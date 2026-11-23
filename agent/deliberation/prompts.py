"""Prompt templates for each deliberation stage."""
from __future__ import annotations

import json

SYSTEM = """You are part of a content team running a social media persona on Instagram Reels and TikTok.
Write in plain language. Be specific and concrete, never generic.

{persona}

{continuity}

{memory}"""

PROPOSER = """ROLE: {role}

TASK TYPE: {task_type} — {task_desc}
REQUEST: {prompt}
{note}{feedback}
Produce {n} distinct options. Make them genuinely different from each other (different angles, not rewordings).
Reply with ONLY JSON in this shape:
{{"options": [{{"text": "<the option itself>", "rationale": "<one sentence: why this works>"}}]}}"""

CRITIQUE = """You are reviewing options written by OTHER team members for this request:
REQUEST ({task_type}): {prompt}

OPTIONS:
{options}

Judge them against this rubric:
{rubric}

Be blunt and specific. Reply with ONLY JSON:
{{"critiques": [{{"id": <option number>, "strengths": ["..."], "weaknesses": ["..."], "issues": ["persona/consistency/policy problems, if any"], "steal_this": "<best element worth keeping, or empty>", "score": <1-10>}}]}}"""

SYNTH = """ROLE: {role}
You are the final writer. Fuse the best material below into the {n} strongest options for:
REQUEST ({task_type}): {prompt}
{note}{feedback}
CANDIDATE OPTIONS:
{options}
{critiques}
Rules: keep the persona voice exact; combine strong elements across candidates; drop weak ideas; options must differ.
Reply with ONLY JSON:
{{"options": [{{"text": "...", "rationale": "..."}}],
  "decision_log": ["<which candidates/critique points you used or rejected, and why>"]}}"""

JUDGE = """You are an impartial judge for a social media persona's content.
REQUEST ({task_type}): {prompt}

Score each option 1-10 against this rubric (10 = would clearly outperform the persona's past content):
{rubric}

OPTIONS:
{options}

Reply with ONLY JSON:
{{"scores": [{{"id": <option number>, "score": <1-10>, "notes": "<one sentence>"}}],
  "feedback": "<what the team should change to make the best option a 10>"}}"""


def fmt_options(options: list[dict]) -> str:
    return "\n".join(
        f"[{i}] {o['text']}" + (f"\n    (why: {o['rationale']})" if o.get("rationale") else "")
        for i, o in enumerate(options, 1)
    )


def fmt_rubric(rubric: list[str]) -> str:
    return "\n".join(f"- {r}" for r in rubric)


def fmt_critiques(critiques: list[dict]) -> str:
    if not critiques:
        return ""
    return "\nCRITIQUES FROM REVIEWERS:\n" + json.dumps(critiques, ensure_ascii=False, indent=1) + "\n"


def fmt_memory(similar: list[dict]) -> str:
    if not similar:
        return ""
    lines = ["PREVIOUSLY APPROVED CONTENT (match this quality; do not repeat it):"]
    for m in similar:
        perf = (m.get("meta") or {}).get("perf")
        tag = f" (real result: {perf})" if perf else ""
        lines.append(f"- [{m['kind']}]{tag} {m['text']}")
    return "\n".join(lines)
