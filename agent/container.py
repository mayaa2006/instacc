"""Wires all components together."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from agent.config import Settings
from agent.db import DB
from agent.deliberation.engine import DeliberationEngine
from agent.deliberation.router import Router
from agent.analytics.service import AnalyticsService
from agent.autopilot import Autopilot
from agent.community.comments import CommentService
from agent.content.posts import PostService
from agent.content.producer import Producer
from agent.evals.harness import EvalService
from agent.memory.continuity import Continuity
from agent.publish.instagram import InstagramClient
from agent.publish.tiktok import TikTokClient
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
    continuity: Continuity
    publishers: dict
    posts: PostService
    comments: CommentService
    analytics: AnalyticsService
    autopilot: Autopilot
    pairing: object = None   # set by the web app (it knows the port)

    @classmethod
    def build(cls, settings: Settings, transport: httpx.AsyncBaseTransport | None = None,
              publish_transport: httpx.AsyncBaseTransport | None = None, sleep=None) -> "Container":
        db = DB(settings.data_path / "instacc.db")
        servers = ServerManager(settings)
        gpu = GpuScheduler(
            db, servers, comfyui_url=settings.agent.comfyui.url,
            free_comfy_on_think=settings.agent.comfyui.free_on_think,
        )
        llm = LLMClient(settings, db, gpu=gpu, transport=transport)
        memory = Memory(db, llm)
        memory.seed_persona(settings.root / "config")
        continuity = Continuity(db, llm)
        engine = DeliberationEngine(settings, db, llm, memory, continuity)
        tasks = TaskService(db, engine, memory, router=Router(settings, db, llm))
        evals = EvalService(settings, db, engine, llm, memory)
        pub = settings.agent.publishing
        kw = {"sleep": sleep} if sleep else {}
        publishers: dict = {}
        if pub.instagram.enabled:
            publishers["instagram"] = InstagramClient(pub.instagram, pub.ai_label, publish_transport, **kw)
        if pub.tiktok.enabled:
            publishers["tiktok"] = TikTokClient(pub.tiktok, db, pub.ai_label, publish_transport, **kw)
        producer = Producer(db, llm, memory, continuity, style=settings.style,
                            style_id=settings.style_file.active if settings.style_file else None)
        posts = PostService(settings, db, tasks, producer, continuity, publishers)
        comments = CommentService(settings, db, llm, memory, publishers)
        analytics = AnalyticsService(settings, db, publishers)
        autopilot = Autopilot(settings, posts, comments, analytics, publishers, gpu)
        return cls(settings, db, servers, gpu, llm, memory, engine, tasks, evals, continuity, publishers,
                   posts, comments, analytics, autopilot)

    async def start(self) -> None:
        await self.gpu.start()
        self.tasks.resume_pending()
        self.autopilot.start()

    async def stop(self) -> None:
        await self.autopilot.stop()
        for client in self.publishers.values():
            await client.aclose()
        await self.gpu.shutdown()
        await self.llm.aclose()
        self.db.close()
