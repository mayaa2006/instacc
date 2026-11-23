"""HTTP API + dashboard. Reach it from your phone over Tailscale."""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent.config import Settings
from agent.container import Container
from agent.tasks import TaskError

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class NewTask(BaseModel):
    type: str
    prompt: str
    note: str | None = None


class Approve(BaseModel):
    index: int  # 0-based


class Rethink(BaseModel):
    note: str | None = None


class GpuMode(BaseModel):
    mode: Literal["think", "render"]
    holder: str | None = None
    ttl_s: float | None = None


def create_app(settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None,
               start_bot: bool = True) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        c = Container.build(settings, transport=transport)
        app.state.c = c
        await c.start()
        bot = None
        if start_bot and settings.secrets.telegram_token:
            from agent.bots.telegram_bot import TelegramBot

            bot = TelegramBot(c)
            await bot.start()
        try:
            yield
        finally:
            if bot:
                await bot.stop()
            await c.stop()

    app = FastAPI(title="instacc agent", lifespan=lifespan)

    def auth(request: Request) -> None:
        token = settings.secrets.api_token
        if not token:
            return
        got = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(got, token):
            raise HTTPException(401, "bad or missing token")

    def c(request: Request) -> Container:
        return request.app.state.c

    def task_or_404(cont: Container, tid: str) -> dict:
        t = cont.db.task_get(tid)
        if not t:
            raise HTTPException(404, "task not found")
        return t

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/manifest.webmanifest")
    def manifest():
        return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")

    @app.get("/health")
    async def health(request: Request):
        cont = c(request)
        out = {}
        async with httpx.AsyncClient(timeout=3) as h:
            for name, ep in settings.agent.endpoints.items():
                if ep.kind != "local":
                    continue
                url = ep.base_url.rstrip("/").removesuffix("/v1") + "/health"
                try:
                    out[name] = (await h.get(url)).status_code == 200
                except httpx.HTTPError:
                    out[name] = False
        return {"ok": True, "local_endpoints": out, "gpu_mode": cont.gpu.mode}

    api = APIRouter(prefix="/api", dependencies=[Depends(auth)])

    @api.get("/status")
    def status(request: Request):
        cont = c(request)
        return {
            "gpu": cont.gpu.status(),
            "budget": {"monthly_usd": settings.agent.budget.monthly_usd,
                       "spent_usd": round(cont.db.cloud_spend_this_month(), 4)},
            "pending": len(cont.db.task_list(status="awaiting_approval")),
        }

    @api.get("/persona")
    def get_persona(request: Request):
        rec = c(request).memory.persona_record()
        if not rec:
            raise HTTPException(404, "no persona yet")
        return rec

    @api.put("/persona")
    def put_persona(body: dict, request: Request):
        try:
            v = c(request).memory.save_persona(body)
        except Exception as e:
            raise HTTPException(422, str(e))
        return {"version": v}

    @api.get("/persona/history")
    def persona_history(request: Request):
        return c(request).db.persona_history()

    @api.get("/task-types")
    def task_types():
        return {k: {"description": v.description, "sensitivity": v.sensitivity}
                for k, v in settings.tasks.tasks.items()}

    @api.post("/tasks")
    async def create_task(body: NewTask, request: Request):
        try:
            tid = c(request).tasks.create(body.type, body.prompt, note=body.note, source="api")
        except (TaskError, ValueError) as e:
            raise HTTPException(422, str(e))
        return {"id": tid}

    @api.get("/tasks")
    def list_tasks(request: Request, status: str | None = None, limit: int = 50):
        return c(request).db.task_list(status=status, limit=limit)

    @api.get("/tasks/{tid}")
    def get_task(tid: str, request: Request):
        cont = c(request)
        t = task_or_404(cont, tid)
        t["trace"] = cont.db.trace_get(tid)
        return t

    @api.post("/tasks/{tid}/approve")
    async def approve(tid: str, body: Approve, request: Request):
        try:
            return await c(request).tasks.approve(tid, body.index)
        except TaskError as e:
            raise HTTPException(409, str(e))

    @api.post("/tasks/{tid}/reject")
    async def reject(tid: str, request: Request):
        try:
            return await c(request).tasks.reject(tid)
        except TaskError as e:
            raise HTTPException(409, str(e))

    @api.post("/tasks/{tid}/rethink")
    async def rethink(tid: str, body: Rethink, request: Request):
        try:
            return {"id": c(request).tasks.rethink(tid, body.note, source="api")}
        except TaskError as e:
            raise HTTPException(409, str(e))

    @api.get("/gpu")
    def gpu_status(request: Request):
        return c(request).gpu.status()

    @api.post("/gpu/mode")
    async def gpu_mode(body: GpuMode, request: Request):
        """Video side: call with mode=render before rendering (blocks until the GPU is free),
        and mode=think when done. Re-call with render to renew a ttl lease."""
        return await c(request).gpu.set_mode(body.mode, holder=body.holder, ttl_s=body.ttl_s)

    app.include_router(api)
    return app
