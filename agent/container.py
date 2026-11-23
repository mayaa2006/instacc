"""Wires all components together."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from agent.config import Settings
from agent.db import DB
from agent.deliberation.engine import DeliberationEngine
from agent.deliberation.router import Router
from agent.evals.harness import EvalService
from agent.memory.store import Memory
from agent.models.client import LLMClient
from agent.scheduler.gpu import GpuScheduler
from agent.scheduler.servers import ServerManager
from agent.tasks import TaskService


@dataclass
class Container:
    settings: Settings
    db: DB
    servers: ServerManager
    gpu: GpuScheduler
    llm: LLMClient
    memory: Memory
    engine: DeliberationEngine
    tasks: TaskService
    evals: EvalService

    @classmethod
    def build(cls, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> "Container":
        db = DB(settings.data_path / "instacc.db")
        servers = ServerManager(settings)
        gpu = GpuScheduler(
            db, servers, comfyui_url=settings.agent.comfyui.url,
            free_comfy_on_think=settings.agent.comfyui.free_on_think,
        )
        llm = LLMClient(settings, db, gpu=gpu, transport=transport)
        memory = Memory(db, llm)
        memory.seed_persona(settings.root / "config")
        engine = DeliberationEngine(settings, db, llm, memory)
        tasks = TaskService(db, engine, memory, router=Router(settings, db, llm))
        evals = EvalService(settings, db, engine, llm, memory)
        return cls(settings, db, servers, gpu, llm, memory, engine, tasks, evals)

    async def start(self) -> None:
        await self.gpu.start()
        self.tasks.resume_pending()

    async def stop(self) -> None:
        await self.gpu.shutdown()
        await self.llm.aclose()
        self.db.close()
