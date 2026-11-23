"""GPU scheduler: the 4090 is either in THINK mode (LLMs on GPU) or RENDER mode
(GPU belongs to video generation, LLMs on CPU / cloud).

With 16 GB system RAM, servers that have no profile for the current mode are
stopped, and any call to them waits until that mode ends (private tasks simply
pause during an overnight render; public ones can still use the cloud).

The video pipeline calls `POST /gpu/mode {"mode": "render", "holder": "video", "ttl_s": 3600}`
before rendering and `{"mode": "think"}` when done. The call returns only once
the LLM servers have left the GPU. A render lease with a TTL reverts to think
automatically if the video side crashes and never releases it (renew by
calling again).
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import AsyncIterator, Literal

import httpx

from agent.db import DB
from agent.scheduler.servers import ServerManager

log = logging.getLogger(__name__)
Mode = Literal["think", "render"]


class GpuScheduler:
    def __init__(self, db: DB, servers: ServerManager, drain_timeout_s: float = 600,
                 comfyui_url: str | None = None, free_comfy_on_think: bool = True):
        self.db = db
        self.servers = servers
        self.drain_timeout_s = drain_timeout_s
        state = db.kv_get("gpu", {}) or {}
        self.mode: Mode = state.get("mode", "think")
        self.holder: str | None = state.get("holder")
        self.since: float = state.get("since", time.time())
        self.expires_at: float | None = state.get("expires_at")
        self._switch_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._ready.set()
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._ttl_task: asyncio.Task | None = None
        self._mode_changed = asyncio.Event()
        self._waiting = 0
        self.comfyui_url = comfyui_url
        self.free_comfy_on_think = free_comfy_on_think

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "holder": self.holder,
            "since": self.since,
            "expires_at": self.expires_at,
            "switching": not self._ready.is_set(),
            "inflight_local_calls": self._inflight,
            "calls_waiting_for_think_mode": self._waiting,
            "servers": self.servers.status(),
        }

    async def start(self) -> None:
        if self.mode == "render" and self.expires_at and self.expires_at < time.time():
            self.mode, self.holder, self.expires_at = "think", None, None
        await self.servers.apply_profile(self.mode)
        self._persist()
        self._arm_ttl()

    async def shutdown(self) -> None:
        if self._ttl_task:
            self._ttl_task.cancel()
        await self.servers.stop_all()

    @contextlib.asynccontextmanager
    async def lease(self, server: str | None = None) -> AsyncIterator[None]:
        """Held by every call to a local model server, so a mode switch can
        wait for in-flight requests instead of killing them. If `server` is
        stopped in the current mode, waits until the mode changes."""
        while True:
            await self._ready.wait()
            if server is None or self.servers.available(server, self.mode):
                break
            self._waiting += 1
            try:
                await self._mode_changed.wait()
            finally:
                self._waiting -= 1
        self._inflight += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._inflight -= 1
            if self._inflight == 0:
                self._idle.set()

    async def set_mode(self, mode: Mode, holder: str | None = None, ttl_s: float | None = None) -> dict:
        async with self._switch_lock:
            if mode == self.mode:
                # renew lease / update holder without restarting anything
                self.holder = holder if mode == "render" else None
                self.expires_at = time.time() + ttl_s if (mode == "render" and ttl_s) else None
                self._persist()
                self._arm_ttl()
                return self.status()
            log.info("GPU mode %s -> %s (holder=%s)", self.mode, mode, holder)
            self._ready.clear()
            try:
                try:
                    await asyncio.wait_for(self._idle.wait(), self.drain_timeout_s)
                except asyncio.TimeoutError:
                    log.warning("drain timeout; switching with %d local calls in flight", self._inflight)
                if mode == "think":
                    await self._free_comfyui()
                await self.servers.apply_profile(mode)
                self.mode = mode
                self.holder = holder if mode == "render" else None
                self.since = time.time()
                self.expires_at = time.time() + ttl_s if (mode == "render" and ttl_s) else None
                self._persist()
            finally:
                self._ready.set()
                changed, self._mode_changed = self._mode_changed, asyncio.Event()
                changed.set()
            self._arm_ttl()
            return self.status()

    async def _free_comfyui(self) -> None:
        if not (self.comfyui_url and self.free_comfy_on_think):
            return
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                await c.post(self.comfyui_url.rstrip("/") + "/free",
                             json={"unload_models": True, "free_memory": True})
            await asyncio.sleep(2)  # give the driver a moment to release VRAM
            log.info("asked ComfyUI to free VRAM")
        except httpx.HTTPError as e:
            log.info("ComfyUI not reachable for /free (%s); continuing", type(e).__name__)

    def _persist(self) -> None:
        self.db.kv_set(
            "gpu", {"mode": self.mode, "holder": self.holder, "since": self.since, "expires_at": self.expires_at}
        )

    def _arm_ttl(self) -> None:
        if self._ttl_task and not self._ttl_task.done() and self._ttl_task is not asyncio.current_task():
            self._ttl_task.cancel()
        self._ttl_task = None
        if self.mode == "render" and self.expires_at:
            self._ttl_task = asyncio.create_task(self._ttl_watch(self.expires_at))

    async def _ttl_watch(self, expires_at: float) -> None:
        await asyncio.sleep(max(0.0, expires_at - time.time()))
        if self.mode == "render" and self.expires_at == expires_at:
            log.warning("render lease held by %s expired; reverting GPU to think mode", self.holder)
            await self.set_mode("think")
