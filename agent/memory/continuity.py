"""Continuity ledger: facts that must stay consistent across videos.

Entries are global (series_id NULL) or per series. Facts extracted automatically from
approved content arrive as `proposed`; only `active` entries are fed to the models,
so a model can't silently rewrite the persona's world.
"""
from __future__ import annotations

import logging
import time
import uuid

from agent.db import DB
from agent.deliberation.jsonutil import extract_json

log = logging.getLogger(__name__)

KINDS = ("outfit", "location", "prop", "storyline", "joke", "fact", "lighting", "relationship")

EXTRACT_PROMPT = """Extract continuity facts from this approved content for a recurring social media character.
A continuity fact is something future videos must stay consistent with: outfits, places, props,
ongoing storylines, running jokes, relationships, established facts about her life.
Only extract facts that are clearly established, not vague vibes. Max 6.

KNOWN FACTS (don't repeat these unless the content changes them):
{known}

CONTENT:
<<<
{text}
>>>

Reply with ONLY JSON:
{{"facts": [{{"kind": "<one of: {kinds}>", "key": "<short stable name, e.g. 'apartment' or 'latte art saga'>", "value": "<the fact>"}}]}}"""


class Continuity:
    def __init__(self, db: DB, llm=None, max_prompt_entries: int = 40):
        self.db = db
        self.llm = llm
        self.max_prompt_entries = max_prompt_entries

    # ------------------------------------------------------------ series
    def create_series(self, name: str, description: str = "") -> str:
        sid = uuid.uuid4().hex[:8]
        self.db.execute("INSERT INTO series(id,name,description,status,created_at) VALUES(?,?,?,?,?)",
                        (sid, name, description, "active", time.time()))
        return sid

    def list_series(self, include_archived: bool = False) -> list[dict]:
        sql = "SELECT * FROM series" + ("" if include_archived else " WHERE status='active'")
        return self.db.query(sql + " ORDER BY created_at DESC")

    def get_series(self, sid: str) -> dict | None:
        rows = self.db.query("SELECT * FROM series WHERE id=?", (sid,))
        return rows[0] if rows else None

    def archive_series(self, sid: str) -> None:
        self.db.execute("UPDATE series SET status='archived' WHERE id=?", (sid,))

    # ----------------------------------------------------------- entries
    def add(self, kind: str, key: str, value: str, *, series_id: str | None = None,
            status: str = "active", source: str | None = None) -> int:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if status not in ("proposed", "active", "retired"):
            raise ValueError("bad status")
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO continuity(series_id,kind,key,value,status,source,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (series_id, kind, key.strip(), value.strip(), status, source, now, now),
        )
        return int(cur.lastrowid)

    def update(self, entry_id: int, **fields) -> dict:
        allowed = {k: v for k, v in fields.items() if k in ("kind", "key", "value", "status", "series_id")}
        if "kind" in allowed and allowed["kind"] not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if allowed:
            allowed["updated_at"] = time.time()
            cols = ", ".join(f"{k}=?" for k in allowed)
            self.db.execute(f"UPDATE continuity SET {cols} WHERE id=?", (*allowed.values(), entry_id))
        return self.get(entry_id)

    def get(self, entry_id: int) -> dict:
        rows = self.db.query("SELECT * FROM continuity WHERE id=?", (entry_id,))
        if not rows:
            raise KeyError(entry_id)
        return rows[0]

    def confirm(self, entry_id: int) -> dict:
        """Activate a proposed entry; it replaces any active entry with the same series/kind/key."""
        e = self.get(entry_id)
        self.db.execute(
            "UPDATE continuity SET status='retired', updated_at=? WHERE status='active' AND kind=? AND key=?"
            " AND COALESCE(series_id,'')=COALESCE(?,'') AND id<>?",
            (time.time(), e["kind"], e["key"], e["series_id"], entry_id),
        )
        return self.update(entry_id, status="active")

    def entries(self, series_id: str | None = None, status: str | None = "active",
                include_global: bool = True) -> list[dict]:
        where, params = [], []
        if status:
            where.append("status=?")
            params.append(status)
        if series_id is not None:
            where.append("(series_id=?" + (" OR series_id IS NULL)" if include_global else ")"))
            params.append(series_id)
        sql = "SELECT * FROM continuity" + (" WHERE " + " AND ".join(where) if where else "")
        return self.db.query(sql + " ORDER BY updated_at DESC", tuple(params))

    def mark_used(self, series_id: str | None, post_id: str) -> None:
        self.db.execute(
            "UPDATE continuity SET last_used_post=? WHERE status='active' AND (series_id=? OR series_id IS NULL)",
            (post_id, series_id),
        )

    # ------------------------------------------------------------ prompts
    def to_prompt(self, series_id: str | None = None) -> str:
        items = self.entries(series_id=series_id) if series_id else self.entries()
        if not items:
            return ""
        names = {s["id"]: s["name"] for s in self.list_series(include_archived=True)}
        lines = ["CONTINUITY LEDGER (established facts; stay consistent, never contradict):"]
        for e in items[: self.max_prompt_entries]:
            scope = f" ({names.get(e['series_id'], e['series_id'])})" if e["series_id"] else ""
            lines.append(f"- [{e['kind']}{scope}] {e['key']}: {e['value']}")
        return "\n".join(lines)

    def for_video_job(self, series_id: str | None) -> dict:
        """Collapse the ledger into VideoJob.continuity fields."""
        out: dict = {"props": []}
        for e in self.entries(series_id=series_id):
            if e["kind"] == "prop":
                out["props"].append(e["value"])
            elif e["kind"] in ("outfit", "location", "lighting") and e["kind"] not in out:
                out[e["kind"]] = e["value"]
        return out

    # -------------------------------------------------------- extraction
    async def extract(self, text: str, *, series_id: str | None = None, source: str | None = None) -> list[int]:
        """Propose ledger entries from approved content (local model; content is private)."""
        if self.llm is None:
            return []
        known = "\n".join(f"- [{e['kind']}] {e['key']}: {e['value']}"
                          for e in self.entries(series_id=series_id)[:40]) or "(none)"
        try:
            res = await self.llm.chat("main", [{"role": "user", "content": EXTRACT_PROMPT.format(
                known=known, text=text[:4000], kinds=", ".join(KINDS))}],
                temperature=0.0, sensitivity="private", json_mode=True, max_tokens=600)
            facts = (extract_json(res.text) or {}).get("facts", [])
        except Exception as e:
            log.warning("continuity extraction failed: %s", e)
            return []
        existing = {(e["kind"], e["key"].lower()): e for e in self.entries(series_id=series_id, status=None)
                    if e["status"] in ("active", "proposed")}
        ids = []
        for f in facts if isinstance(facts, list) else []:
            try:
                kind, key, value = str(f["kind"]).lower(), str(f["key"]).strip(), str(f["value"]).strip()
            except (KeyError, TypeError):
                continue
            if kind not in KINDS or not key or not value:
                continue
            prev = existing.get((kind, key.lower()))
            if prev and prev["value"].strip().lower() == value.lower():
                continue
            ids.append(self.add(kind, key, value, series_id=series_id, status="proposed", source=source))
        return ids
