"""Router: the fast local model picks a task type for free-form requests ("auto")."""
from __future__ import annotations

import logging
import re

from agent.config import Settings
from agent.db import DB
from agent.deliberation.jsonutil import extract_json
from agent.models.client import LLMClient

log = logging.getLogger(__name__)

ROUTER_PROMPT = """Classify this request for a social media persona's content team.
Pick exactly one task type:
{types}

Guidance: pick the cheapest type that does the job well. Long-term or concept-level requests
need deep thinking; tiny or throwaway requests use the fast path.

REQUEST: {prompt}

Reply with ONLY JSON: {{"type": "<one of: {names}>", "reason": "<short>"}}"""

# Keyword fallback if the fast model is unavailable or answers nonsense.
_KEYWORDS = [
    (r"\bhooks?\b|\bopening line|first (?:second|line)", "hooks"),
    (r"\bcaptions?\b|hashtags?", "caption"),
    (r"\bideas?\b|\bconcepts?\b|\bseries\b|\bstrategy\b|\bplan\b", "idea"),
]


class Router:
    def __init__(self, settings: Settings, db: DB, llm: LLMClient, slot: str = "fast"):
        self.s = settings
        self.db = db
        self.llm = llm
        self.slot = slot

    def choices(self) -> dict[str, str]:
        return {k: v.description for k, v in self.s.tasks.tasks.items()}

    def fallback(self, prompt: str) -> str:
        names = self.choices()
        for rx, t in _KEYWORDS:
            if t in names and re.search(rx, prompt, re.IGNORECASE):
                return t
        return self.s.tasks.default_type if self.s.tasks.default_type in names else next(iter(names))

    async def route(self, task_id: str, prompt: str) -> tuple[str, str]:
        choices = self.choices()
        user = ROUTER_PROMPT.format(
            types="\n".join(f"- {k}: {v}" for k, v in choices.items()),
            names=", ".join(choices), prompt=prompt,
        )
        try:
            res = await self.llm.chat(self.slot, [{"role": "user", "content": user}], temperature=0.0,
                                      sensitivity="private", json_mode=True, max_tokens=400)
            obj = extract_json(res.text)
            t = str(obj.get("type", "")).strip()
            reason = str(obj.get("reason", ""))
            self.db.trace_add(task_id, "route", slot=res.slot, model=res.model, endpoint=res.endpoint,
                              output=res.text, latency_ms=res.latency_ms, cost_usd=res.cost_usd)
            if t in choices:
                return t, reason
            log.info("router returned unknown type %r; using keyword fallback", t)
        except Exception as e:
            log.warning("router failed (%s); using keyword fallback", e)
        t = self.fallback(prompt)
        self.db.trace_add(task_id, "route", note=f"keyword fallback -> {t}")
        return t, "keyword fallback"
