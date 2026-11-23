"""Deliberation engine: proposers -> optional cross-critique -> synthesizer -> judge, with feedback loops.
Every LLM call is written to the trace.

Variety tools (see config/tasks.yaml), most useful when every slot is the same local model:
role-specific sampling + per-call seeds, random creative constraints, sequential anti-repetition,
embedding de-duplication, and a shuffled audience-panel judge."""
from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field

from agent.config import RoleCfg, Settings, TaskCfg
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
    sampling: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, s: str) -> "Spec":
        slot, _, rest = s.partition("@")
        temp, _, role = rest.partition(":")
        return cls(slot=slot.strip(), temperature=float(temp) if temp else 0.7, role=role.strip() or None)


class DeliberationEngine:
    def __init__(self, settings: Settings, db: DB, llm: LLMClient, memory: Memory, continuity=None):
        self.s = settings
        self.db = db
        self.llm = llm
        self.memory = memory
        self.continuity = continuity

    def task_cfg(self, task_type: str) -> TaskCfg:
        if task_type not in self.s.tasks.tasks:
            raise ValueError(f"unknown task type '{task_type}'. Known: {', '.join(self.s.tasks.tasks)}")
        return self.s.tasks.tasks[task_type]

    def role_cfg(self, role: str | None) -> RoleCfg | None:
        r = self.s.tasks.roles.get(role) if role else None
        if isinstance(r, str):
            return RoleCfg(prompt=r)
        return r

    def role_text(self, role: str | None) -> str:
        if not role:
            return "You are a sharp, experienced content creator."
        rc = self.role_cfg(role)
        return rc.prompt if rc else role

    def proposer_spec(self, raw: str, rng: random.Random) -> Spec:
        """Spec with the role's own temperature/sampling and a fresh seed, so the same model explores differently."""
        sp = Spec.parse(raw)
        rc = self.role_cfg(sp.role)
        if rc:
            if rc.temperature is not None:
                sp.temperature = rc.temperature
            sp.sampling = rc.sampling()
        sp.sampling["seed"] = rng.randrange(1, 2**31 - 1)
        return sp

    def _is_gpu_serial(self, slot: str) -> bool:
        """Calls to the GPU model run one at a time anyway, so 'sequential' costs nothing for them."""
        ep = self.s.agent.endpoints[self.s.agent.slots[slot].endpoint]
        return ep.kind == "local" and ep.uses_gpu

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
                    sampling={k: v for k, v in spec.sampling.items() if not k.startswith("_")} or None,
                )
            except Exception as e:
                log.warning("%s %s failed: %s", stage, spec.slot, e)
                self.db.trace_add(task_id, stage, loop=loop, slot=spec.slot, role=spec.role,
                                  temperature=spec.temperature, note=f"error: {e}")
                return None, None
            notes = [f"rerouted: {res.rerouted}"] if res.rerouted else []
            if spec.sampling.get("_note"):
                notes.append(spec.sampling["_note"])
            note = "; ".join(notes) or None
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
            continuity=self.continuity.to_prompt() if self.continuity else "",
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

            # 1. proposers
            rng = random.Random(f"{task_id}:{loop}")
            specs = [self.proposer_spec(p, rng) for p in cfg.proposers]
            pool_c = list(self.s.tasks.creative_constraints)
            rng.shuffle(pool_c)

            def prompt_for(i: int, sp: Spec, taken: list[dict]) -> str:
                constraint = ""
                if cfg.constraints and pool_c:
                    c = pool_c[i % len(pool_c)]
                    constraint = f"CREATIVE CONSTRAINT FOR YOU (follow it): {c}\n"
                    sp.sampling["_note"] = f"constraint: {c}"
                return P.PROPOSER.format(role=self.role_text(sp.role), note=note, feedback=fb, constraint=constraint,
                                         taken=P.fmt_taken(taken), **base)

            results: list = [None] * len(specs)
            if cfg.diversify == "sequential":
                # GPU-model proposers go one by one, each told which angles are taken; CPU/cloud ones run alongside
                side = {i: asyncio.create_task(self._ask(task_id, loop, "propose", sp, system, prompt_for(i, sp, []), cfg))
                        for i, sp in enumerate(specs) if not self._is_gpu_serial(sp.slot)}
                taken: list[dict] = []
                for i, sp in enumerate(specs):
                    if i in side:
                        continue
                    for j, t in side.items():   # also avoid whatever the side proposers already finished
                        if t.done() and results[j] is None:
                            results[j] = t.result()
                            taken += normalize_options(results[j][1]) if results[j][1] is not None else []
                    results[i] = await self._ask(task_id, loop, "propose", sp, system, prompt_for(i, sp, taken), cfg)
                    taken += normalize_options(results[i][1]) if results[i][1] is not None else []
                for j, t in side.items():
                    if results[j] is None:
                        results[j] = await t
            else:
                results = list(await asyncio.gather(*(
                    self._ask(task_id, loop, "propose", sp, system, prompt_for(i, sp, []), cfg)
                    for i, sp in enumerate(specs))))
            pooled: list[dict] = []
            owners: list[int] = []
            for idx, (_, obj) in enumerate(results):
                opts = normalize_options(obj) if obj is not None else []
                pooled += opts
                owners += [idx] * len(opts)
            if not pooled:
                raise RuntimeError("all proposers failed; check that model servers are running (GET /health)")

            # 1b. drop near-duplicates (same idea, different words)
            if cfg.dedupe and len(pooled) > 2:
                pooled, owners = await self._dedupe(task_id, loop, pooled, owners, cfg.dedupe)

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

            # 4. judge (optionally as an audience panel, on shuffled options)
            if not cfg.judge:
                break
            judge_summary = await self._judge(task_id, loop, cfg, system, task, rubric, final, rng)
            scored = [o["score"] for o in final if isinstance(o.get("score"), (int, float))]
            best = max(scored) if scored else None
            if best is None or best >= cfg.threshold:
                break
            feedback = judge_summary or "Scores were below the bar; be bolder and more specific."

        for o in final:
            o.pop("_scores", None)
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

    # ------------------------------------------------------------- variety
    async def _dedupe(self, task_id: str, loop: int, pooled: list[dict], owners: list[int],
                      threshold: float) -> tuple[list[dict], list[int]]:
        try:
            vecs = await self.llm.embed([o["text"] for o in pooled])
        except Exception as e:
            log.info("dedupe skipped: %s", e)
            return pooled, owners

        def cos(a, b):
            na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
            return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0

        keep: list[int] = []
        for i, v in enumerate(vecs):
            if all(cos(v, vecs[k]) < threshold for k in keep):
                keep.append(i)
        if len(keep) < len(pooled):
            self.db.trace_add(task_id, "dedupe", loop=loop,
                              note=f"kept {len(keep)} of {len(pooled)} (dropped near-duplicates, cos >= {threshold})",
                              output="\n".join(pooled[i]["text"] for i in range(len(pooled)) if i not in keep))
        return [pooled[i] for i in keep], [owners[i] for i in keep]

    async def _judge(self, task_id: str, loop: int, cfg: TaskCfg, system: str, task: dict, rubric: str,
                     final: list[dict], rng: random.Random) -> str:
        personas = list(self.s.tasks.judge_personas)
        rng.shuffle(personas)
        panel = personas[: max(1, cfg.judge_panel)] if cfg.judge_panel > 1 and personas else [None]
        for o in final:
            o["_scores"] = []
        feedback = []
        for persona in panel:
            order = list(range(len(final)))
            if cfg.shuffle_for_judge:
                rng.shuffle(order)
            shown = [final[i] for i in order]
            line = (f"Score as this viewer would react: {persona}. Stay critical; most content is a 5-6.\n"
                    if persona else "")
            spec = Spec(cfg.judge, 0.0, sampling={"seed": rng.randrange(1, 2**31 - 1),
                                                  **({"_note": f"panel: {persona}"} if persona else {})})
            _, obj = await self._ask(
                task_id, loop, "judge", spec, system,
                P.JUDGE.format(task_type=task["type"], prompt=task["prompt"], rubric=rubric,
                               options=P.fmt_options(shown), persona_line=line),
                cfg,
            )
            if not isinstance(obj, dict):
                continue
            for sc in obj.get("scores", []) or []:
                try:
                    pos = int(sc.get("id")) - 1
                    if 0 <= pos < len(order):
                        o = final[order[pos]]
                        o["_scores"].append(float(sc.get("score")))
                        o.setdefault("judge_notes", str(sc.get("notes", "")))
                except (TypeError, ValueError):
                    continue
            if obj.get("feedback"):
                feedback.append(str(obj["feedback"]))
        for o in final:
            if o["_scores"]:
                o["score"] = round(sum(o["_scores"]) / len(o["_scores"]), 2)
                if len(o["_scores"]) > 1:
                    o["panel_scores"] = o["_scores"]
        return " | ".join(feedback)
