"""Background loop: publish due posts, poll comments, collect metrics, keep tokens fresh."""
from __future__ import annotations

import asyncio
import logging
import time

log = logging.getLogger(__name__)


class Autopilot:
    def __init__(self, settings, posts, comments, analytics, publishers: dict, gpu=None):
        self.s = settings
        self.cfg = settings.agent.autopilot
        self.posts = posts
        self.comments = comments
        self.analytics = analytics
        self.publishers = publishers
        self.gpu = gpu
        self._task: asyncio.Task | None = None
        self._last_comments = 0.0
        self._last_token = 0.0
        self.last_tick: float | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        if self.cfg.enabled and self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    def status(self) -> dict:
        return {"enabled": self.cfg.enabled, "running": bool(self._task and not self._task.done()),
                "last_tick": self.last_tick, "last_error": self.last_error,
                "platforms": {k: getattr(v, "configured", lambda: True)() for k, v in self.publishers.items()}}

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("autopilot tick failed")
                self.last_error = str(e)
            await asyncio.sleep(self.cfg.tick_s)

    async def tick(self, now: float | None = None) -> dict:
        now = now or time.time()
        self.last_tick = now
        done = {"published": 0, "comments": 0, "metrics": 0}
        for p in self.posts.due():
            try:
                r = await self.posts.publish(p["id"])
                done["published"] += r["status"] == "published"
            except Exception as e:
                log.warning("publish %s failed: %s", p["id"], e)

        window = now - self.cfg.comment_window_days * 86400
        recent = [p for p in self.posts.list(status="published", limit=200)
                  if any((v or {}).get("at", 0) >= window for v in (p.get("published") or {}).values())]

        if now - self._last_comments >= self.cfg.comment_poll_min * 60 and "instagram" in self.publishers:
            self._last_comments = now
            ig = self.publishers["instagram"]
            if getattr(ig, "configured", lambda: True)():
                for p in recent:
                    info = (p.get("published") or {}).get("instagram") or {}
                    if info.get("id"):
                        try:
                            raw = await ig.comments(info["id"])
                            done["comments"] += len(self.comments.ingest("instagram", p["id"], info["id"], raw))
                        except Exception as e:
                            log.warning("comments for %s failed: %s", p["id"], e)
                if done["comments"]:
                    await self.comments.process_new()

        if recent or self.posts.list(status="published", limit=1):
            done["metrics"] = await self.analytics.collect(self.posts.list(status="published", limit=500))

        tt = self.publishers.get("tiktok")
        if tt is not None and now - self._last_token > 7 * 86400 and tt.configured():
            self._last_token = now
            try:
                await tt.access_token()   # refreshes if close to expiry; keeps the rotating token alive
            except Exception as e:
                log.warning("tiktok token refresh failed: %s", e)
        return done
