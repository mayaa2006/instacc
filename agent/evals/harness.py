"""Eval harness: does the deliberation pipeline actually beat a single model?

A run executes every golden case through every *variant*, keeps each variant's top
option, then compares variants pairwise and blind:
  - humans vote in the dashboard / Telegram (/rate), never seeing which is which;
  - an LLM judge votes twice per pair with the order swapped; disagreement = tie
    (cancels position bias).

Variants:
  pipeline                        the task type's own config
  pipeline:no-critique,no-loops   ablations: no-critique | no-judge | no-synth | no-loops
  single:<slot>                   one call, one option, no deliberation (e.g. single:main, single:judge)
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import random
import secrets
import time
import uuid
from pathlib import Path

import yaml

from agent.config import Settings, TaskCfg
from agent.db import DB
from agent.deliberation.engine import DeliberationEngine
from agent.deliberation.jsonutil import extract_json
from agent.models.client import LLMClient

log = logging.getLogger(__name__)

PAIR_PROMPT = """You are judging two pieces of content for the same social media persona and request.
Ignore length and formatting; judge which would perform better with the persona's audience
(hook strength, persona fit, specificity, novelty, platform fit).

PERSONA:
{persona}

REQUEST ({task_type}): {prompt}

OPTION 1:
{a}

OPTION 2:
{b}

Reply with ONLY JSON: {{"winner": "1" | "2" | "tie", "reason": "<one sentence>"}}"""

ABLATIONS = {"no-critique", "no-judge", "no-synth", "no-loops"}


class EvalError(ValueError):
    pass


def load_cases(root: Path, name: str = "golden") -> list[dict]:
    path = root / "evals" / f"{name}.yaml"
    if not path.exists():
        raise EvalError(f"eval set {path} not found")
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]
    for c in cases:
        if not {"id", "type", "prompt"} <= set(c):
            raise EvalError(f"bad case {c}: needs id, type, prompt")
    return cases


def variant_cfg(variant: str, base: TaskCfg, slots: set[str]) -> TaskCfg:
    kind, _, arg = variant.partition(":")
    if kind == "single":
        if arg not in slots:
            raise EvalError(f"unknown slot in variant {variant!r}")
        return TaskCfg(description=base.description, sensitivity="public", n_options=1,
                       proposers=[f"{arg}@0.7:writer"], critique=False, synthesizer=None,
                       judge=None, threshold=0, max_loops=0)
    if kind == "pipeline":
        mods = {m for m in arg.split(",") if m}
        if bad := mods - ABLATIONS:
            raise EvalError(f"unknown ablation(s) {bad}; known: {sorted(ABLATIONS)}")
        cfg = base.model_copy(deep=True)
        cfg.sensitivity = "public"  # golden cases are synthetic, cloud allowed
        if "no-critique" in mods:
            cfg.critique = False
        if "no-synth" in mods:
            cfg.synthesizer = None
        if "no-judge" in mods:
            cfg.judge = None
        if "no-loops" in mods:
            cfg.max_loops = 0
        return cfg
    raise EvalError(f"unknown variant {variant!r} (use pipeline[:mods] or single:<slot>)")


class EvalService:
    def __init__(self, settings: Settings, db: DB, engine: DeliberationEngine, llm: LLMClient, memory):
        self.s = settings
        self.db = db
        self.engine = engine
        self.llm = llm
        self.memory = memory
        self._tokens: dict[str, dict] = {}   # blind pair token -> pair info (server-side only)
        self._running: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ runs
    def start(self, variants: list[str], *, name: str | None = None, cases: str | list[dict] = "golden",
              limit: int | None = None, types: list[str] | None = None, auto_judge: str | None = "judge",
              concurrency: int = 1) -> str:
        if len(variants) < 2:
            raise EvalError("need at least two variants to compare")
        case_list = load_cases(self.s.root, cases) if isinstance(cases, str) else cases
        if types:
            case_list = [c for c in case_list if c["type"] in types]
        if limit:
            case_list = case_list[:limit]
        if not case_list:
            raise EvalError("no cases selected")
        slots = set(self.s.agent.slots)
        for c in case_list:
            base = self.engine.task_cfg(c["type"])
            for v in variants:
                variant_cfg(v, base, slots)  # validate up front
        if auto_judge and auto_judge not in slots:
            raise EvalError(f"unknown judge slot {auto_judge}")
        rid = uuid.uuid4().hex[:8]
        config = {"variants": variants, "cases": case_list, "auto_judge": auto_judge, "concurrency": concurrency}
        self.db.execute(
            "INSERT INTO eval_runs(id,name,config,status,progress,created_at) VALUES(?,?,?,?,?,?)",
            (rid, name or f"{' vs '.join(variants)}", json.dumps(config), "running",
             json.dumps({"done": 0, "total": len(case_list) * len(variants)}), time.time()),
        )
        t = asyncio.create_task(self._run(rid))
        self._running.add(t)
        t.add_done_callback(self._running.discard)
        return rid

    async def wait_idle(self) -> None:
        while self._running:
            await asyncio.gather(*list(self._running), return_exceptions=True)

    def get_run(self, rid: str) -> dict:
        rows = self.db.query("SELECT * FROM eval_runs WHERE id=?", (rid,))
        if not rows:
            raise EvalError(f"eval run {rid} not found")
        r = rows[0]
        r["config"] = json.loads(r["config"])
        r["progress"] = json.loads(r["progress"] or "{}")
        return r

    def list_runs(self, limit: int = 20) -> list[dict]:
        rows = self.db.query("SELECT id,name,status,progress,created_at,error FROM eval_runs "
                             "ORDER BY created_at DESC LIMIT ?", (limit,))
        for r in rows:
            r["progress"] = json.loads(r["progress"] or "{}")
        return rows

    async def _run(self, rid: str) -> None:
        run = self.get_run(rid)
        cfg = run["config"]
        sem = asyncio.Semaphore(max(1, int(cfg.get("concurrency", 1))))
        done = 0
        total = len(cfg["cases"]) * len(cfg["variants"])

        async def one(case: dict, variant: str) -> None:
            nonlocal done
            async with sem:
                base = self.engine.task_cfg(case["type"])
                vcfg = variant_cfg(variant, base, set(self.s.agent.slots))
                tid = self.db.task_create(case["type"], case["prompt"], source=f"eval:{rid}")
                self.db.task_update(tid, status="eval")
                t0 = time.monotonic()
                output, error = None, None
                try:
                    result = await self.engine.run(tid, vcfg)
                    opts = result.get("options") or []
                    output = opts[0]["text"] if opts else None
                    if output is None:
                        error = "no options"
                    self.db.task_update(tid, result=result)
                except Exception as e:
                    log.warning("eval %s case %s variant %s failed: %s", rid, case["id"], variant, e)
                    error = str(e)
                wall = time.monotonic() - t0
                cost = self.db.query("SELECT COALESCE(SUM(cost_usd),0) AS c FROM trace_steps WHERE task_id=?",
                                     (tid,))[0]["c"]
                self.db.execute(
                    "INSERT OR REPLACE INTO eval_items(run_id,case_id,task_type,prompt,variant,task_id,output,"
                    "cost_usd,wall_s,error) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (rid, case["id"], case["type"], case["prompt"], variant, tid, output, cost, wall, error),
                )
                done += 1
                self.db.execute("UPDATE eval_runs SET progress=? WHERE id=?",
                                (json.dumps({"done": done, "total": total, "stage": "generating"}), rid))

        try:
            await asyncio.gather(*(one(c, v) for c in cfg["cases"] for v in cfg["variants"]))
            if cfg.get("auto_judge"):
                await self._auto_judge(rid, cfg["auto_judge"])
            self.db.execute("UPDATE eval_runs SET status='done', progress=? WHERE id=?",
                            (json.dumps({"done": total, "total": total, "stage": "done"}), rid))
        except Exception as e:
            log.exception("eval run %s failed", rid)
            self.db.execute("UPDATE eval_runs SET status='failed', error=? WHERE id=?", (str(e), rid))

    # ----------------------------------------------------------------- pairs
    def _items(self, rid: str) -> dict[str, dict[str, dict]]:
        by_case: dict[str, dict[str, dict]] = {}
        for it in self.db.query("SELECT * FROM eval_items WHERE run_id=?", (rid,)):
            by_case.setdefault(it["case_id"], {})[it["variant"]] = it
        return by_case

    def _pairs(self, rid: str) -> list[tuple[str, str, str]]:
        variants = self.get_run(rid)["config"]["variants"]
        out = []
        for case_id, items in self._items(rid).items():
            for a, b in itertools.combinations(variants, 2):
                if items.get(a, {}).get("output") and items.get(b, {}).get("output"):
                    out.append((case_id, a, b))
        return out

    def _vote(self, rid: str, case_id: str, a: str, b: str, winner: str, voter: str, reason: str = "") -> None:
        self.db.execute(
            "INSERT INTO eval_votes(run_id,case_id,a,b,winner,voter,reason,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (rid, case_id, a, b, winner, voter, reason, time.time()),
        )

    async def _compare_once(self, slot: str, persona: str, item: dict, first: str, second: str) -> str:
        user = PAIR_PROMPT.format(persona=persona, task_type=item["task_type"], prompt=item["prompt"],
                                  a=first, b=second)
        res = await self.llm.chat(slot, [{"role": "user", "content": user}], temperature=0.0,
                                  sensitivity="public", json_mode=True, max_tokens=200)
        try:
            w = str(extract_json(res.text).get("winner", "tie")).strip().lower()
        except ValueError:
            w = "tie"
        return w if w in ("1", "2") else "tie"

    async def _auto_judge(self, rid: str, slot: str) -> None:
        persona = self.memory.persona()
        ptext = persona.to_prompt() if persona else "(none)"
        items = self._items(rid)
        pairs = self._pairs(rid)
        for n, (case_id, a, b) in enumerate(pairs, 1):
            ia, ib = items[case_id][a], items[case_id][b]
            if ia["output"].strip() == ib["output"].strip():
                self._vote(rid, case_id, a, b, "tie", f"llm:{slot}", "identical")
                continue
            try:
                ab = await self._compare_once(slot, ptext, ia, ia["output"], ib["output"])
                ba = await self._compare_once(slot, ptext, ia, ib["output"], ia["output"])
            except Exception as e:
                log.warning("auto-judge failed on %s %s/%s: %s", case_id, a, b, e)
                continue
            # consistent only if the same *content* wins in both orders
            if ab == "1" and ba == "2":
                winner = a
            elif ab == "2" and ba == "1":
                winner = b
            else:
                winner = "tie"
            self._vote(rid, case_id, a, b, winner, f"llm:{slot}", f"orders: {ab}/{ba}")
            self.db.execute("UPDATE eval_runs SET progress=? WHERE id=?",
                            (json.dumps({"done": n, "total": len(pairs), "stage": "judging"}), rid))

    # ------------------------------------------------------------ human votes
    def next_pair(self, rid: str) -> dict | None:
        """A random unvoted pair, sides shuffled. The client only gets a token."""
        voted = {(v["case_id"], v["a"], v["b"]) for v in
                 self.db.query("SELECT case_id,a,b FROM eval_votes WHERE run_id=? AND voter='human'", (rid,))}
        todo = [p for p in self._pairs(rid) if p not in voted]
        if not todo:
            return None
        case_id, a, b = random.choice(todo)
        items = self._items(rid)[case_id]
        left, right = (a, b) if random.random() < 0.5 else (b, a)
        token = secrets.token_urlsafe(8)
        self._tokens[token] = {"run_id": rid, "case_id": case_id, "a": a, "b": b, "left": left, "right": right}
        if len(self._tokens) > 500:
            self._tokens.pop(next(iter(self._tokens)))
        return {
            "token": token, "task_type": items[a]["task_type"], "prompt": items[a]["prompt"],
            "left": items[left]["output"], "right": items[right]["output"], "remaining": len(todo),
        }

    def vote(self, token: str, choice: str) -> None:
        info = self._tokens.pop(token, None)
        if not info:
            raise EvalError("unknown or already-used pair token")
        if choice not in ("left", "right", "tie"):
            raise EvalError("choice must be left, right or tie")
        winner = "tie" if choice == "tie" else info[choice]
        self._vote(info["run_id"], info["case_id"], info["a"], info["b"], winner, "human")

    # ---------------------------------------------------------------- report
    def report(self, rid: str, voter: str | None = None) -> dict:
        run = self.get_run(rid)
        variants = run["config"]["variants"]
        items = self.db.query("SELECT * FROM eval_items WHERE run_id=?", (rid,))
        votes = self.db.query("SELECT * FROM eval_votes WHERE run_id=?", (rid,))
        if voter == "human":
            votes = [v for v in votes if v["voter"] == "human"]
        elif voter == "llm":
            votes = [v for v in votes if v["voter"].startswith("llm:")]

        stats = {v: {"wins": 0, "losses": 0, "ties": 0} for v in variants}
        h2h = {a: {b: {"wins": 0, "losses": 0, "ties": 0} for b in variants if b != a} for a in variants}
        for v in votes:
            a, b, w = v["a"], v["b"], v["winner"]
            if w == "tie":
                for x, y in ((a, b), (b, a)):
                    stats[x]["ties"] += 1
                    h2h[x][y]["ties"] += 1
            else:
                loser = b if w == a else a
                stats[w]["wins"] += 1
                stats[loser]["losses"] += 1
                h2h[w][loser]["wins"] += 1
                h2h[loser][w]["losses"] += 1

        def rate(d):
            n = d["wins"] + d["losses"] + d["ties"]
            return round((d["wins"] + 0.5 * d["ties"]) / n, 3) if n else None

        rows = []
        for v in variants:
            its = [i for i in items if i["variant"] == v]
            ok = [i for i in its if i["output"]]
            rows.append({
                "variant": v, **stats[v], "win_rate": rate(stats[v]),
                "avg_cost_usd": round(sum(i["cost_usd"] or 0 for i in ok) / len(ok), 5) if ok else None,
                "avg_wall_s": round(sum(i["wall_s"] or 0 for i in ok) / len(ok), 2) if ok else None,
                "failures": len(its) - len(ok),
            })
        rows.sort(key=lambda r: (r["win_rate"] is None, -(r["win_rate"] or 0)))
        head = {a: {b: {**d, "win_rate": rate(d)} for b, d in row.items()} for a, row in h2h.items()}
        by_type: dict[str, dict[str, dict]] = {}
        type_of = {i["case_id"]: i["task_type"] for i in items}
        for v in votes:
            t = type_of.get(v["case_id"], "?")
            for x in (v["a"], v["b"]):
                d = by_type.setdefault(t, {}).setdefault(x, {"wins": 0, "losses": 0, "ties": 0})
                d["ties" if v["winner"] == "tie" else ("wins" if v["winner"] == x else "losses")] += 1
        by_type_rates = {t: {x: rate(d) for x, d in m.items()} for t, m in by_type.items()}
        return {"run": {k: run[k] for k in ("id", "name", "status", "progress", "created_at", "error")},
                "votes": len(votes), "variants": rows, "head_to_head": head, "by_task_type": by_type_rates}
