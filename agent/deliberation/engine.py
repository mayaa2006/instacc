"""Deliberation engine (Phase 1): independent proposers -> optional cross-critique ->
synthesizer -> judge, with feedback loops. Every LLM call is written to the trace."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from agent.config import Settings, TaskCfg
from agent.db import DB
from agent.deliberation import prompts as P
from agent.deliberation.jsonutil import extract_json, normalize_options
from agent.memory.store import Memory
from agent.models.client import ChatResult, LLMClient

log = logging.getLogger(__name__)


@dataclass
class Spec:
    slot: str
    temperature: float = 0.7
    role: str | None = None

    @classmethod
    def parse(cls, s: str) -> "Spec":
        slot, _, rest = s.partition("@")
        temp, _, role = rest.partition(":")
        return cls(slot=slot.strip(), temperature=float(temp) if temp else 0.7, role=role.strip() or None)


class DeliberationEngine:
    def __init__(self, settings: Settings, db: DB, llm: LLMClient, memory: Memory):
        self.s = settings
        self.db = db
        self.llm = llm
        self.memory = memory

    def task_cfg(self, task_type: str) -> TaskCfg:
        if task_type not in self.s.tasks.tasks:
            raise ValueError(f"unknown task type '{task_type}'. Known: {', '.join(self.s.tasks.tasks)}")
        return self.s.tasks.tasks[task_type]

    def role_text(self, role: str | None) -> str:
        if not role:
            return "You are a sharp, experienced content creator."
        return self.s.tasks.roles.get(role, role)

    # ---------------------------------------------------------------- llm call
    async def _ask(
        self, task_id: str, loop: int, stage: str, spec: Spec, system: str, user: str, cfg: TaskCfg,
    ) -> tuple[ChatResult | None, object | None]:
        """One LLM call that must return JSON; retries once with a repair prompt."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(2):
            try:
                res = await self.llm.chat(
                    spec.slot, messages, temperature=spec.temperature, sensitivity=cfg.sensitivity, json_mode=True,
                )
            except Exception as e:
                log.warning("%s %s failed: %s", stage, spec.slot, e)
                self.db.trace_add(task_id, stage, loop=loop, slot=spec.slot, role=spec.role,
                                  temperature=spec.temperature, note=f"error: {e}")
                return None, None
            note = f"rerouted: {res.rerouted}" if res.rerouted else None
            self.db.trace_add(
                task_id, stage, loop=loop, slot=res.slot, model=res.model, endpoint=res.endpoint, role=spec.role,
                temperature=spec.temperature, output=res.text, latency_ms=res.latency_ms, cost_usd=res.cost_usd,
                note=note,
            )
            try:
                return res, extract_json(res.text)
            except ValueError:
                messages += [
                    {"role": "assistant", "content": res.text},
                    {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object."},
                ]
        return None, None

    # --------------------------------------------------------------------- run
    async def run(self, task_id: str, cfg: TaskCfg | None = None) -> dict:
        """Run deliberation for a stored task. `cfg` overrides the task type's config
        (used by the eval harness to run alternative pipelines on the same request)."""
        task = self.db.task_get(task_id)
        if not task:
            raise ValueError(f"task {task_id} not found")
        cfg = cfg or self.task_cfg(task["type"])
        persona = self.memory.persona()
        similar = await self.memory.similar(task["prompt"], k=5)
        system = P.SYSTEM.format(
            persona=persona.to_prompt() if persona else "PERSONA: (not set yet)",
            memory=P.fmt_memory(similar),
        ).strip()
        rubric = P.fmt_rubric(self.s.tasks.rubric)
        note = f"EXTRA DIRECTION FROM THE OWNER: {task['note']}\n" if task.get("note") else ""
        base = dict(task_type=task["type"], task_desc=cfg.description, prompt=task["prompt"], n=cfg.n_options)

        feedback = ""
        final: list[dict] = []
        decision_log: list[str] = []
        judge_summary = ""
        best = None
        loops_run = 0

        for loop in range(cfg.max_loops + 1):
            loops_run = loop + 1
            fb = f"JUDGE FEEDBACK ON THE PREVIOUS ROUND (fix this): {feedback}\n" if feedback else ""

            # 1. independent proposers (parallel, they never see each other)
            specs = [Spec.parse(p) for p in cfg.proposers]
            results = await asyncio.gather(*(
                self._ask(task_id, loop, "propose", sp, system,
                          P.PROPOSER.format(role=self.role_text(sp.role), note=note, feedback=fb, **base), cfg)
                for sp in specs
            ))
            pooled: list[dict] = []
            owners: list[int] = []
            for idx, (_, obj) in enumerate(results):
                opts = normalize_options(obj) if obj is not None else []
                pooled += opts
                owners += [idx] * len(opts)
            if not pooled:
                raise RuntimeError("all proposers failed; check that model servers are running (GET /health)")

            # 2. cross-critique: each proposer reviews the options it did NOT write
            critiques: list[dict] = []
            if cfg.critique and len(specs) > 1:
                async def critique(i: int, sp: Spec):
                    others = [(n, o) for n, (o, ow) in enumerate(zip(pooled, owners), 1) if ow != i]
                    if not others:
                        return []
                    text = "\n".join(f"[{n}] {o['text']}" for n, o in others)
                    _, obj = await self._ask(
                        task_id, loop, "critique", Spec(sp.slot, 0.3, sp.role), system,
                        P.CRITIQUE.format(task_type=task["type"], prompt=task["prompt"], options=text, rubric=rubric),
                        cfg,
                    )
                    return (obj or {}).get("critiques", []) if isinstance(obj, dict) else []
                for cs in await asyncio.gather(*(critique(i, sp) for i, sp in enumerate(specs))):
                    critiques += cs

            # 3. synthesizer
            if cfg.synthesizer:
                sp = Spec.parse(cfg.synthesizer)
                _, obj = await self._ask(
                    task_id, loop, "synthesize", sp, system,
                    P.SYNTH.format(role=self.role_text(sp.role), note=note, feedback=fb,
                                   options=P.fmt_options(pooled), critiques=P.fmt_critiques(critiques), **base),
                    cfg,
                )
                final = normalize_options(obj)[: cfg.n_options] if obj is not None else []
                if isinstance(obj, dict) and isinstance(obj.get("decision_log"), list):
                    decision_log = [str(x) for x in obj["decision_log"]]
            if not final:
                final = pooled[: cfg.n_options]

            # 4. judge
            if not cfg.judge:
                break
            _, obj = await self._ask(
                task_id, loop, "judge", Spec(cfg.judge, 0.0), system,
                P.JUDGE.format(task_type=task["type"], prompt=task["prompt"], rubric=rubric,
                               options=P.fmt_options(final)),
                cfg,
            )
            if isinstance(obj, dict):
                for s in obj.get("scores", []) or []:
                    try:
                        i = int(s.get("id")) - 1
                        if 0 <= i < len(final):
                            final[i]["score"] = float(s.get("score"))
                            final[i]["judge_notes"] = str(s.get("notes", ""))
                    except (TypeError, ValueError):
                        continue
                judge_summary = str(obj.get("feedback", ""))
            scored = [o["score"] for o in final if isinstance(o.get("score"), (int, float))]
            best = max(scored) if scored else None
            if best is None or best >= cfg.threshold:
                break
            feedback = judge_summary or "Scores were below the bar; be bolder and more specific."

        if any("score" in o for o in final):
            final.sort(key=lambda o: o.get("score", 0), reverse=True)
        return {
            "options": final,
            "decision_log": decision_log,
            "judge_feedback": judge_summary,
            "best_score": best,
            "threshold": cfg.threshold if cfg.judge else None,
            "below_threshold": bool(cfg.judge and best is not None and best < cfg.threshold),
            "loops": loops_run,
        }
