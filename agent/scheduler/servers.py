"""Starts/stops llama.cpp `llama-server` processes with mode-specific profiles."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from agent.config import Settings

log = logging.getLogger(__name__)


@dataclass
class _Proc:
    popen: subprocess.Popen
    profile: str
    log_file: object = field(repr=False, default=None)


class ServerManager:
    def __init__(self, settings: Settings):
        self.s = settings
        self.cfg = settings.agent.llama_server
        self.procs: dict[str, _Proc] = {}
        self.log_dir = settings.data_path / "logs"

    @property
    def enabled(self) -> bool:
        return self.cfg.manage_processes

    def binary(self) -> str:
        b = self.cfg.binary
        if "/" in b or "\\" in b:
            return str(self.s.resolve(b))
        return b

    def available(self, name: str, mode: str) -> bool:
        """Whether server `name` runs in `mode` (always true if we don't manage processes)."""
        if not self.enabled or name not in self.s.agent.servers:
            return True
        return mode in self.s.agent.servers[name].profiles

    def command(self, name: str, profile: str) -> list[str]:
        srv = self.s.agent.servers[name]
        args = srv.profiles.get(profile, [])
        return [
            self.binary(), "-m", str(self.s.resolve(srv.model_path)),
            "--host", srv.host, "--port", str(srv.port), *args,
        ]

    def status(self) -> dict[str, dict]:
        out = {}
        for name in self.s.agent.servers:
            p = self.procs.get(name)
            out[name] = {
                "running": bool(p and p.popen.poll() is None),
                "profile": p.profile if p else None,
                "pid": p.popen.pid if p else None,
            }
        return out

    async def apply_profile(self, profile: str) -> None:
        """Ensure every server runs with `profile`; restart only those that differ."""
        if not self.enabled:
            return
        for name, srv in self.s.agent.servers.items():
            cur = self.procs.get(name)
            if cur and cur.popen.poll() is None and cur.profile == profile:
                continue
            if cur:
                await self.stop(name)
            if profile not in srv.profiles:
                log.info("server %s has no '%s' profile; leaving it stopped", name, profile)
                continue
            await self.start(name, profile)
        await asyncio.gather(*(self.wait_ready(n) for n in self.procs))

    async def start(self, name: str, profile: str) -> None:
        srv = self.s.agent.servers[name]
        model = self.s.resolve(srv.model_path)
        if not model.exists():
            log.warning("model for server %s not found at %s; skipping (see docs/SETUP.md)", name, model)
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        lf = open(self.log_dir / f"{name}.log", "ab")
        cmd = self.command(name, profile)
        log.info("starting %s: %s", name, " ".join(cmd))
        try:
            popen = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        except FileNotFoundError:
            lf.close()
            log.error("llama-server binary not found: %s (set llama_server.binary)", self.binary())
            return
        self.procs[name] = _Proc(popen=popen, profile=profile, log_file=lf)

    async def stop(self, name: str) -> None:
        p = self.procs.pop(name, None)
        if not p:
            return
        if p.popen.poll() is None:
            p.popen.terminate()
            try:
                await asyncio.to_thread(p.popen.wait, 20)
            except subprocess.TimeoutExpired:
                p.popen.kill()
                await asyncio.to_thread(p.popen.wait, 10)
        if p.log_file:
            p.log_file.close()
        log.info("stopped %s", name)

    async def stop_all(self) -> None:
        for name in list(self.procs):
            await self.stop(name)

    async def wait_ready(self, name: str) -> bool:
        p = self.procs.get(name)
        if not p:
            return False
        srv = self.s.agent.servers[name]
        url = f"http://{srv.host}:{srv.port}/health"
        deadline = time.monotonic() + self.cfg.startup_timeout_s
        async with httpx.AsyncClient(timeout=5) as client:
            while time.monotonic() < deadline:
                if p.popen.poll() is not None:
                    log.error("server %s exited with code %s; see %s", name, p.popen.returncode,
                              Path(self.log_dir) / f"{name}.log")
                    return False
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        log.info("server %s ready", name)
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
        log.error("server %s not ready after %ss", name, self.cfg.startup_timeout_s)
        return False
