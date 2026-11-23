"""Task lifecycle: queued -> running -> awaiting_approval -> approved / rejected / superseded (rethink) / failed."""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from agent.db import DB
from agent.deliberation.engine import DeliberationEngine
from agent.memory.store import Memory

log = logging.getLogger(__name__)
Listener = Callable[[str, dict], Awaitable[None]]   # (event, task)


class TaskError(ValueError):
    pass


AUTO = "auto"


class TaskService:
    def __init__(self, db: DB, engine: DeliberationEngine, memory: Memory, max_concurrent: int = 2,
                 router=None):
        self.db = db
        self.engine = engine
        self.memory = memory
        self.router = router
        self.listeners: list[Listener] = []
        self._sem = asyncio.Semaphore(max_concurrent)
        self._running: set[asyncio.Task] = set()

    def on_event(self, fn: Listener) -> None:
        self.listeners.append(fn)

    async def _emit(self, event: str, task: dict) -> None:
        for fn in self.listeners:
            try:
                await fn(event, task)
            except Exception:
                log.exception("listener failed for %s", event)

    def create(self, type_: str, prompt: str, *, note: str | None = None, parent_id: str | None = None,
               source: str | None = None) -> str:
        if type_ != AUTO or self.router is None:
            self.engine.task_cfg(type_)  # validate type
        if not prompt.strip():
            raise TaskError("prompt is empty")
        tid = self.db.task_create(type_, prompt.strip(), note=note, parent_id=parent_id, source=source)
        self._spawn(tid)
        return tid

    def _spawn(self, tid: str) -> None:
        t = asyncio.create_task(self._run(tid))
        self._running.add(t)
        t.add_done_callback(self._running.discard)

    async def _run(self, tid: str) -> None:
        async with self._sem:
            self.db.task_update(tid, status="running")
            try:
                task = self.db.task_get(tid)
                if task["type"] == AUTO and self.router is not None:
                    chosen, reason = await self.router.route(tid, task["prompt"])
                    self.db.task_update(tid, type=chosen)
                    log.info("task %s routed to %s (%s)", tid, chosen, reason)
                result = await self.engine.run(tid)
                self.db.task_update(tid, status="awaiting_approval", result=result, error=None)
                await self._emit("ready", self.db.task_get(tid))
            except Exception as e:
                log.exception("task %s failed", tid)
                self.db.task_update(tid, status="failed", error=str(e))
                await self._emit("failed", self.db.task_get(tid))

    def resume_pending(self) -> None:
        """Re-run tasks interrupted by a restart."""
        for status in ("queued", "running"):
            for t in self.db.task_list(status=status, limit=100):
                self._spawn(t["id"])

    async def wait_idle(self) -> None:
        while self._running:
            await asyncio.gather(*list(self._running), return_exceptions=True)

    def _get_pending(self, tid: str) -> dict:
        task = self.db.task_get(tid)
        if not task:
            raise TaskError(f"task {tid} not found")
        if task["status"] != "awaiting_approval":
            raise TaskError(f"task {tid} is {task['status']}, not awaiting approval")
        return task

    async def approve(self, tid: str, index: int) -> dict:
        task = self._get_pending(tid)
        options = (task["result"] or {}).get("options", [])
        if not 0 <= index < len(options):
            raise TaskError(f"option {index + 1} does not exist")
        chosen = options[index]
        self.db.task_update(tid, status="approved", chosen=chosen)
        await self.memory.remember(
            task["type"], chosen["text"],
            {"task_id": tid, "prompt": task["prompt"], "score": chosen.get("score")},
        )
        task = self.db.task_get(tid)
        await self._emit("approved", task)
        return task

    async def approve_many(self, tid: str, indices: list[int]) -> dict:
        """Approve several options at once (e.g. a week plan). `chosen` becomes a list."""
        task = self._get_pending(tid)
        options = (task["result"] or {}).get("options", [])
        idx = sorted(set(indices))
        if not idx or any(not 0 <= i < len(options) for i in idx):
            raise TaskError("pick at least one existing option")
        chosen = [options[i] for i in idx]
        self.db.task_update(tid, status="approved", chosen=chosen)
        for o in chosen:
            await self.memory.remember(task["type"], o["text"],
                                       {"task_id": tid, "prompt": task["prompt"], "score": o.get("score")})
        task = self.db.task_get(tid)
        await self._emit("approved", task)
        return task

    async def reject(self, tid: str) -> dict:
        self._get_pending(tid)
        self.db.task_update(tid, status="rejected")
        task = self.db.task_get(tid)
        await self._emit("rejected", task)
        return task

    def rethink(self, tid: str, note: str | None = None, source: str | None = None) -> str:
        task = self._get_pending(tid)
        merged = "; ".join(x for x in (task.get("note"), note) if x) or None
        if (task.get("source") or "").startswith("post:"):
            source = task["source"]  # keep the rethought caption attached to its post
        self.db.task_update(tid, status="superseded")
        return self.create(task["type"], task["prompt"], note=merged, parent_id=tid, source=source)
