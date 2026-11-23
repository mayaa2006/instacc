"""Post lifecycle and calendar.

planned -> scripting -> script_review -> render_queued -> rendering -> captioning -> caption_review
        -> scheduled -> publishing -> published | failed        (dropped at any point)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable

from agent.config import Settings
from agent.content.producer import Producer, job_json
from agent.db import DB
from agent.publish.base import AuthError, PublishError

log = logging.getLogger(__name__)
Listener = Callable[[str, dict], Awaitable[None]]

ACTIVE = ("planned", "scripting", "script_review", "render_queued", "rendering", "captioning",
          "caption_review", "scheduled", "publishing")


class PostError(ValueError):
    pass


def _row(r: dict) -> dict:
    for k in ("platforms", "video_job", "published"):
        if r.get(k):
            r[k] = json.loads(r[k])
    r["ai_label"] = bool(r.get("ai_label"))
    return r


class PostService:
    def __init__(self, settings: Settings, db: DB, tasks, producer: Producer, continuity, publishers: dict):
        self.s = settings
        self.pub_cfg = settings.agent.publishing
        self.db = db
        self.tasks = tasks
        self.producer = producer
        self.continuity = continuity
        self.publishers = publishers   # platform -> client
        self.listeners: list[Listener] = []
        self._bg: set[asyncio.Task] = set()
        tasks.on_event(self._on_task_event)

    def on_event(self, fn: Listener) -> None:
        self.listeners.append(fn)

    async def _emit(self, event: str, post: dict) -> None:
        for fn in self.listeners:
            try:
                await fn(event, post)
            except Exception:
                log.exception("post listener failed for %s", event)

    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    async def wait_idle(self) -> None:
        while self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)

    # ---------------------------------------------------------------- crud
    def get(self, pid: str) -> dict:
        rows = self.db.query("SELECT * FROM posts WHERE id=?", (pid,))
        if not rows:
            raise PostError(f"post {pid} not found")
        return _row(rows[0])

    def list(self, status: str | None = None, active_only: bool = False, limit: int = 100) -> list[dict]:
        if status:
            rows = self.db.query("SELECT * FROM posts WHERE status=? ORDER BY COALESCE(scheduled_at, created_at)"
                                 " LIMIT ?", (status, limit))
        elif active_only:
            q = ",".join("?" * len(ACTIVE))
            rows = self.db.query(f"SELECT * FROM posts WHERE status IN ({q}) ORDER BY COALESCE(scheduled_at, created_at)"
                                 " LIMIT ?", (*ACTIVE, limit))
        else:
            rows = self.db.query("SELECT * FROM posts ORDER BY COALESCE(scheduled_at, created_at) DESC LIMIT ?",
                                 (limit,))
        return [_row(r) for r in rows]

    def _update(self, pid: str, **f) -> dict:
        for k in ("platforms", "video_job", "published"):
            if k in f and f[k] is not None and not isinstance(f[k], str):
                f[k] = json.dumps(f[k])
        f["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in f)
        self.db.execute(f"UPDATE posts SET {cols} WHERE id=?", (*f.values(), pid))
        return self.get(pid)

    def enabled_platforms(self) -> list[str]:
        out = []
        if self.pub_cfg.instagram.enabled:
            out.append("instagram")
        if self.pub_cfg.tiktok.enabled:
            out.append("tiktok")
        return out

    # ------------------------------------------------------------ calendar
    def next_slots(self, n: int, *, after: float | None = None, lead_h: float = 20) -> list[float]:
        """Next n free posting slots (local time) at least lead_h from now (time to render)."""
        start = max(after or 0, time.time() + lead_h * 3600)
        taken = {round(r["scheduled_at"]) for r in self.db.query(
            "SELECT scheduled_at FROM posts WHERE scheduled_at IS NOT NULL AND status NOT IN ('dropped','failed')")}
        times = sorted(self.pub_cfg.posting_times) or ["18:00"]
        out: list[float] = []
        day = datetime.fromtimestamp(start).replace(hour=0, minute=0, second=0, microsecond=0)
        for _ in range(366):
            for t in times:
                hh, mm = (int(x) for x in t.split(":"))
                ts = day.replace(hour=hh, minute=mm).timestamp()
                if ts >= start and round(ts) not in taken:
                    out.append(ts)
                    if len(out) == n:
                        return out
            day += timedelta(days=1)
        return out

    def create(self, concept: str, *, series_id: str | None = None, scheduled_at: float | None = None,
               idea_task_id: str | None = None, idea_score: float | None = None,
               platforms: list[str] | None = None, auto_script: bool = True) -> dict:
        if not concept.strip():
            raise PostError("concept is empty")
        pid = uuid.uuid4().hex[:8]
        now = time.time()
        self.db.execute(
            "INSERT INTO posts(id,series_id,status,concept,idea_task_id,idea_score,platforms,scheduled_at,ai_label,"
            "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (pid, series_id, "planned", concept.strip(), idea_task_id, idea_score,
             json.dumps(platforms if platforms is not None else self.enabled_platforms()),
             scheduled_at if scheduled_at is not None else self.next_slots(1)[0],
             int(self.pub_cfg.ai_label), now, now),
        )
        post = self.get(pid)
        if auto_script:
            self._spawn(self.generate_script(pid))
        return post

    def create_from_options(self, task: dict, options: list[dict], series_id: str | None = None) -> list[dict]:
        slots = self.next_slots(len(options))
        return [self.create(o["text"], series_id=series_id, scheduled_at=slots[i] if i < len(slots) else None,
                            idea_task_id=task["id"], idea_score=o.get("score"))
                for i, o in enumerate(options)]

    def reschedule(self, pid: str, scheduled_at: float) -> dict:
        p = self.get(pid)
        if p["status"] in ("publishing", "published", "dropped"):
            raise PostError(f"post is {p['status']}")
        return self._update(pid, scheduled_at=scheduled_at)

    def drop(self, pid: str) -> dict:
        p = self.get(pid)
        if p["status"] in ("publishing", "published"):
            raise PostError(f"post is {p['status']}")
        return self._update(pid, status="dropped")

    # -------------------------------------------------------------- script
    async def generate_script(self, pid: str, note: str | None = None) -> dict:
        p = self.get(pid)
        if p["status"] not in ("planned", "script_review", "failed"):
            raise PostError(f"post is {p['status']}")
        self._update(pid, status="scripting", error=None)
        try:
            job = await self.producer.video_job(p, note)
        except Exception as e:
            log.exception("script for %s failed", pid)
            p = self._update(pid, status="failed", error=f"script: {e}")
            await self._emit("failed", p)
            return p
        p = self._update(pid, status="script_review", video_job=job_json(job))
        await self._emit("script_ready", p)
        return p

    def redo_script(self, pid: str, note: str | None = None) -> None:
        p = self.get(pid)
        if p["status"] not in ("script_review", "failed", "planned"):
            raise PostError(f"post is {p['status']}")
        self._spawn(self.generate_script(pid, note))

    async def approve_script(self, pid: str) -> dict:
        p = self.get(pid)
        if p["status"] != "script_review":
            raise PostError(f"post is {p['status']}, not script_review")
        p = self._update(pid, status="render_queued")
        job = p["video_job"] or {}
        text = json.dumps({"concept": job.get("concept"), "continuity": job.get("continuity"),
                           "shots": [s.get("action") for s in job.get("shots", [])]}, ensure_ascii=False)
        self._spawn(self.continuity.extract(text, series_id=p.get("series_id"), source=f"post:{pid}"))
        await self._emit("render_queued", p)
        return p

    # -------------------------------------------------- video side (render)
    def render_queue(self) -> list[dict]:
        return self.list(status="render_queued")

    def claim(self, pid: str, worker: str = "video") -> dict:
        p = self.get(pid)
        if p["status"] != "render_queued":
            raise PostError(f"post is {p['status']}, not render_queued")
        return self._update(pid, status="rendering", error=None)

    async def rendered(self, pid: str, media_path: str) -> dict:
        p = self.get(pid)
        if p["status"] not in ("rendering", "render_queued", "caption_review"):
            raise PostError(f"post is {p['status']}")
        path = Path(media_path)
        if not path.is_absolute():
            path = self.s.root / path
        if not path.exists():
            raise PostError(f"media not found: {path}")
        p = self._update(pid, status="captioning", media_path=str(path))
        job = p.get("video_job") or {}
        prompt = (f"{p['concept']}\n\nVideo: " + "; ".join(s.get("action", "") for s in job.get("shots", []))
                  + "\nInclude 3-6 relevant hashtags at the end.")
        tid = self.tasks.create("caption", prompt, source=f"post:{pid}")
        p = self._update(pid, caption_task_id=tid)
        await self._emit("rendered", p)
        return p

    def render_failed(self, pid: str, error: str) -> dict:
        p = self.get(pid)
        return self._update(pid, status="failed", error=f"render: {error}")

    # ------------------------------------------------------------- captions
    async def _on_task_event(self, event: str, task: dict) -> None:
        src = task.get("source") or ""
        if event == "approved" and task["type"] == "plan" and not src.startswith("eval:"):
            chosen = task["chosen"] if isinstance(task["chosen"], list) else [task["chosen"]]
            posts = self.create_from_options(task, chosen, series_id=self.db.kv_get(f"plan_series:{task['id']}"))
            for p in posts:
                await self._emit("planned", p)
            return
        if not src.startswith("post:"):
            return
        pid = src.split(":", 1)[1]
        try:
            p = self.get(pid)
        except PostError:
            return
        if event == "ready":
            self._update(pid, status="caption_review", caption_task_id=task["id"])
            if self.s.agent.autonomy.publish == "auto" and not (task.get("result") or {}).get("below_threshold"):
                await self.tasks.approve(task["id"], 0)
            else:
                await self._emit("caption_review", self.get(pid))
        elif event == "approved":
            caption = task["chosen"]["text"]
            footer = self.pub_cfg.caption_footer
            if footer and footer not in caption:
                caption = f"{caption}\n\n{footer}"
            p = self._update(pid, caption=caption, status="scheduled")
            if p["scheduled_at"] is None or p["scheduled_at"] < time.time() - 3600:
                p = self._update(pid, scheduled_at=time.time())
            await self._emit("scheduled", p)
        elif event == "failed":
            await self._emit("failed", self._update(pid, status="failed", error=f"caption: {task.get('error')}"))

    # ------------------------------------------------------------- publish
    def due(self) -> list[dict]:
        return [_row(r) for r in self.db.query(
            "SELECT * FROM posts WHERE status='scheduled' AND scheduled_at<=? ORDER BY scheduled_at", (time.time(),))]

    async def publish(self, pid: str) -> dict:
        p = self.get(pid)
        if p["status"] != "scheduled":
            raise PostError(f"post is {p['status']}, not scheduled")
        if not p.get("media_path") or not p.get("caption"):
            raise PostError("post needs media and an approved caption")
        p = self._update(pid, status="publishing")
        results = dict(p.get("published") or {})
        errors = []
        for platform in p["platforms"]:
            if (results.get(platform) or {}).get("state") in ("published", "in_drafts", "private"):
                continue
            client = self.publishers.get(platform)
            if client is None:
                errors.append(f"{platform}: not configured")
                continue
            try:
                if platform == "instagram":
                    r = await client.publish_reel(Path(p["media_path"]), p["caption"])
                else:
                    r = await client.publish_video(Path(p["media_path"]), p["caption"])
                results[platform] = {"state": r.state, "id": r.external_id, "url": r.url, "at": time.time(), **r.extra}
            except AuthError as e:
                errors.append(f"{platform}: {e}")
                await self._emit("needs_reconnect", {**p, "error": str(e), "platform": platform})
            except (PublishError, OSError) as e:
                errors.append(f"{platform}: {e}")
        ok = any((v or {}).get("state") in ("published", "in_drafts", "private") for v in results.values())
        p = self._update(pid, published=results, status="published" if ok else "failed",
                         error="; ".join(errors) or None)
        if ok:
            self.continuity.mark_used(p.get("series_id"), pid)
        await self._emit("published" if ok else "failed", p)
        return p

    def published_since(self, since: float) -> list[dict]:
        return [_row(r) for r in self.db.query(
            "SELECT * FROM posts WHERE status='published' AND updated_at>=? ORDER BY updated_at DESC", (since,))]
