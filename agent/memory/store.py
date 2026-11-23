"""Memory: Persona Bible (versioned) + approved outputs with embeddings for retrieval.

Phase 1 uses brute-force cosine search in Python (fine for thousands of items);
swap to sqlite-vec / pgvector when it grows.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import yaml

from agent.db import DB
from shared.schemas import Persona

log = logging.getLogger(__name__)


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Memory:
    def __init__(self, db: DB, llm=None):
        self.db = db
        self.llm = llm

    # ---------------------------------------------------------------- persona
    def persona(self) -> Persona | None:
        row = self.db.persona_latest()
        return Persona.model_validate(row["data"]) if row else None

    def persona_record(self) -> dict | None:
        return self.db.persona_latest()

    def save_persona(self, data: dict) -> int:
        persona = Persona.model_validate(data)  # validates
        return self.db.persona_save(persona.model_dump(mode="json"))

    def seed_persona(self, config_dir: Path) -> None:
        if self.db.persona_latest():
            return
        for name in ("persona.yaml", "persona.example.yaml"):
            p = config_dir / name
            if p.exists():
                self.save_persona(yaml.safe_load(p.read_text(encoding="utf-8")))
                log.info("seeded persona from %s", p)
                return

    # ------------------------------------------------------------ approved
    async def remember(self, kind: str, text: str, meta: dict | None = None) -> int:
        emb = None
        if self.llm is not None:
            try:
                emb = (await self.llm.embed([text]))[0]
            except Exception as e:  # embed server down: store without vector
                log.warning("embedding failed, storing without vector: %s", e)
        return self.db.memory_add(kind, text, meta, emb)

    async def similar(self, query: str, kind: str | None = None, k: int = 5) -> list[dict]:
        items = self.db.memory_list(kind)
        if not items:
            return []
        qemb = None
        if self.llm is not None:
            try:
                qemb = (await self.llm.embed([query]))[0]
            except Exception as e:
                log.debug("query embed failed: %s", e)
        if qemb is None:
            return items[:k]  # most recent fallback
        scored = [(_cos(qemb, it["embedding"]), it) for it in items if it.get("embedding")]
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [it | {"score": round(s, 3)} for s, it in scored[:k]]
        return out or items[:k]
