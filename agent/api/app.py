"""HTTP API + dashboard. Reach it from your phone over Tailscale."""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import html
import time
import logging
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from agent.config import Settings
from agent.container import Container
from agent.content.posts import PostError
from agent.evals.harness import EvalError
from agent.publish.base import PublishError
from agent.tasks import AUTO, TaskError

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


class NewEval(BaseModel):
    variants: list[str] = ["single:main", "pipeline"]
    name: str | None = None
    cases: str = "golden"
    limit: int | None = None
    types: list[str] | None = None
    auto_judge: str | None = "judge"
    concurrency: int = 1


class Vote(BaseModel):
    token: str
    choice: Literal["left", "right", "tie"]


class ApproveMany(BaseModel):
    indices: list[int]
    series_id: str | None = None


class NewPost(BaseModel):
    concept: str
    series_id: str | None = None
    scheduled_at: float | None = None
    platforms: list[str] | None = None


class Note(BaseModel):
    note: str | None = None


class Rendered(BaseModel):
    media_path: str


class RenderFailed(BaseModel):
    error: str


class Reschedule(BaseModel):
    scheduled_at: float


class ReplyEdit(BaseModel):
    text: str | None = None


class NewSeries(BaseModel):
    name: str
    description: str = ""


class LedgerEntry(BaseModel):
    kind: str
    key: str
    value: str
    series_id: str | None = None
    status: str = "active"


class LedgerPatch(BaseModel):
    kind: str | None = None
    key: str | None = None
    value: str | None = None
    status: str | None = None


class Code(BaseModel):
    code: str


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
        got = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or \
            request.query_params.get("token", "")
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

    @app.get("/classic")
    def classic():
        return FileResponse(STATIC / "classic.html")

    @app.get("/icon.svg")
    def icon():
        return FileResponse(STATIC / "icon.svg", media_type="image/svg+xml")

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

    @api.get("/home")
    def home(request: Request):
        cont = c(request)
        upcoming = cont.posts.list(active_only=True)
        nxt = next((p for p in upcoming if p["status"] == "scheduled"), None)
        published = cont.posts.list(status="published", limit=5)
        return {
            "inbox": len(build_inbox(cont)),
            "next_post": nxt,
            "upcoming": len(upcoming),
            "in_production": sum(p["status"] in ("scripting", "render_queued", "rendering", "captioning")
                                 for p in upcoming),
            "thinking": [{"id": t["id"], "type": t["type"], "prompt": t["prompt"]}
                         for t in cont.db.task_list(status="running", limit=10)
                         + cont.db.task_list(status="queued", limit=10)],
            "recent": published,
            "gpu": cont.gpu.mode,
            "budget": {"spent": round(cont.db.cloud_spend_this_month(), 2),
                       "limit": settings.agent.budget.monthly_usd},
            "verdict": cont.analytics.calibration()["verdict"],
            "platforms": cont.posts.enabled_platforms(),
        }

    def build_inbox(cont) -> list[dict]:
        items: list[dict] = []
        for t in cont.db.task_list(status="awaiting_approval", limit=50):
            if (t.get("source") or "").startswith("post:"):
                continue  # shown with its post as a caption item
            items.append({"kind": "plan" if t["type"] == "plan" else "pick", "id": t["id"], "at": t["updated_at"],
                          "title": t["prompt"], "type": t["type"], "note": t.get("note"),
                          "options": (t["result"] or {}).get("options", []),
                          "below_threshold": (t["result"] or {}).get("below_threshold")})
        for p in cont.posts.list(status="script_review"):
            items.append({"kind": "script", "id": p["id"], "at": p["updated_at"], "title": p["concept"],
                          "scheduled_at": p["scheduled_at"], "job": p["video_job"]})
        for p in cont.posts.list(status="caption_review"):
            t = cont.db.task_get(p["caption_task_id"]) if p.get("caption_task_id") else None
            if not t or t["status"] != "awaiting_approval":
                continue
            items.append({"kind": "caption", "id": p["id"], "task_id": t["id"], "at": p["updated_at"],
                          "title": p["concept"], "scheduled_at": p["scheduled_at"],
                          "has_media": bool(p.get("media_path")),
                          "options": (t["result"] or {}).get("options", [])})
        for p in cont.posts.list(status="failed"):
            items.append({"kind": "failed", "id": p["id"], "at": p["updated_at"], "title": p["concept"],
                          "error": p.get("error")})
        for cm in cont.comments.list(status="drafted", limit=50):
            items.append({"kind": "comment", "id": cm["id"], "at": cm["posted_at"] or cm["created_at"],
                          "title": cm["text"], "username": cm["username"], "risk": cm["risk"],
                          "category": cm["category"], "draft": cm["draft"]})
        for e in cont.continuity.entries(status="proposed"):
            items.append({"kind": "fact", "id": e["id"], "at": e["created_at"], "title": f"{e['key']}: {e['value']}",
                          "fact_kind": e["kind"]})
        order = {"failed": 0, "caption": 1, "script": 2, "plan": 3, "pick": 4, "comment": 5, "fact": 6}
        items.sort(key=lambda i: (order[i["kind"]], -(i["at"] or 0)))
        return items

    @api.get("/inbox")
    def inbox(request: Request):
        return build_inbox(c(request))

    @api.post("/posts/{pid}/retry")
    async def retry_post(pid: str, request: Request):
        return post_call(c(request).posts.retry, pid)

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
        out = {AUTO: {"description": "Let the router pick the task type", "sensitivity": "private"}}
        out.update({k: {"description": v.description, "sensitivity": v.sensitivity}
                    for k, v in settings.tasks.tasks.items()})
        return out

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

    @api.post("/tasks/{tid}/approve-many")
    async def approve_many(tid: str, body: ApproveMany, request: Request):
        cont = c(request)
        if body.series_id:
            cont.db.kv_set(f"plan_series:{tid}", body.series_id)
        try:
            return await cont.tasks.approve_many(tid, body.indices)
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

    # ---- posts / calendar ------------------------------------------------
    def post_call(fn, *a):
        try:
            return fn(*a)
        except PostError as e:
            raise HTTPException(409, str(e))

    @api.get("/posts")
    def list_posts(request: Request, status: str | None = None, active: bool = False):
        return c(request).posts.list(status=status, active_only=active)

    @api.post("/posts")
    async def create_post(body: NewPost, request: Request):
        return post_call(lambda: c(request).posts.create(body.concept, series_id=body.series_id,
                                                         scheduled_at=body.scheduled_at, platforms=body.platforms))

    @api.get("/posts/{pid}")
    def get_post(pid: str, request: Request):
        return post_call(c(request).posts.get, pid)

    @api.get("/posts/{pid}/media")
    def post_media(pid: str, request: Request):
        p = post_call(c(request).posts.get, pid)
        if not p.get("media_path") or not Path(p["media_path"]).exists():
            raise HTTPException(404, "no media yet")
        return FileResponse(p["media_path"])

    @api.post("/posts/{pid}/script/approve")
    async def approve_script(pid: str, request: Request):
        try:
            return await c(request).posts.approve_script(pid)
        except PostError as e:
            raise HTTPException(409, str(e))

    @api.post("/posts/{pid}/script/redo")
    async def redo_script(pid: str, body: Note, request: Request):
        post_call(c(request).posts.redo_script, pid, body.note)
        return {"ok": True}

    @api.post("/posts/{pid}/reschedule")
    def reschedule(pid: str, body: Reschedule, request: Request):
        return post_call(c(request).posts.reschedule, pid, body.scheduled_at)

    @api.post("/posts/{pid}/drop")
    def drop_post(pid: str, request: Request):
        return post_call(c(request).posts.drop, pid)

    @api.post("/posts/{pid}/publish-now")
    async def publish_now(pid: str, request: Request):
        cont = c(request)
        p = post_call(cont.posts.get, pid)
        if p["status"] != "scheduled":
            raise HTTPException(409, f"post is {p['status']}")
        cont.posts._update(pid, scheduled_at=time.time())
        try:
            return await cont.posts.publish(pid)
        except PostError as e:
            raise HTTPException(409, str(e))

    # video side
    @api.get("/render-queue")
    def render_queue(request: Request):
        return c(request).posts.render_queue()

    @api.post("/posts/{pid}/claim")
    def claim(pid: str, request: Request):
        return post_call(c(request).posts.claim, pid)

    @api.post("/posts/{pid}/rendered")
    async def rendered(pid: str, body: Rendered, request: Request):
        try:
            return await c(request).posts.rendered(pid, body.media_path)
        except PostError as e:
            raise HTTPException(409, str(e))

    @api.post("/posts/{pid}/render-failed")
    def render_failed(pid: str, body: RenderFailed, request: Request):
        return post_call(c(request).posts.render_failed, pid, body.error)

    # ---- continuity -----------------------------------------------------
    @api.get("/series")
    def list_series(request: Request):
        return c(request).continuity.list_series()

    @api.post("/series")
    def new_series(body: NewSeries, request: Request):
        return {"id": c(request).continuity.create_series(body.name, body.description)}

    @api.get("/continuity")
    def list_ledger(request: Request, series_id: str | None = None, status: str | None = None):
        return c(request).continuity.entries(series_id=series_id, status=status)

    @api.post("/continuity")
    def add_ledger(body: LedgerEntry, request: Request):
        try:
            return {"id": c(request).continuity.add(body.kind, body.key, body.value, series_id=body.series_id,
                                                    status=body.status, source="manual")}
        except ValueError as e:
            raise HTTPException(422, str(e))

    @api.patch("/continuity/{eid}")
    def patch_ledger(eid: int, body: LedgerPatch, request: Request):
        try:
            return c(request).continuity.update(eid, **body.model_dump(exclude_none=True))
        except (ValueError, KeyError) as e:
            raise HTTPException(422, str(e))

    @api.post("/continuity/{eid}/confirm")
    def confirm_ledger(eid: int, request: Request):
        try:
            return c(request).continuity.confirm(eid)
        except KeyError:
            raise HTTPException(404, "not found")

    # ---- comments -------------------------------------------------------
    @api.get("/comments")
    def list_comments(request: Request, status: str | None = None):
        return c(request).comments.list(status=status)

    @api.post("/comments/{cid}/send")
    async def send_comment(cid: int, body: ReplyEdit, request: Request):
        try:
            return await c(request).comments.send(cid, body.text)
        except (ValueError, KeyError) as e:
            raise HTTPException(409, str(e))

    @api.post("/comments/{cid}/ignore")
    def ignore_comment(cid: int, request: Request):
        return c(request).comments.ignore(cid)

    @api.post("/comments/poll")
    async def poll_comments(request: Request):
        cont = c(request)
        cont.autopilot._last_comments = 0
        return await cont.autopilot.tick()

    # ---- analytics ------------------------------------------------------
    @api.get("/analytics")
    def analytics(request: Request, checkpoint: str = "24h"):
        a = c(request).analytics
        return {"posts": a.table(checkpoint), "calibration": a.calibration(checkpoint)}

    # ---- accounts -------------------------------------------------------
    @api.get("/accounts")
    def accounts(request: Request):
        cont = c(request)
        pub = settings.agent.publishing
        out = {"instagram": {"enabled": pub.instagram.enabled, "login": pub.instagram.login},
               "tiktok": {"enabled": pub.tiktok.enabled, "mode": pub.tiktok.mode},
               "autopilot": cont.autopilot.status()}
        if "instagram" in cont.publishers:
            out["instagram"]["connected"] = cont.publishers["instagram"].configured()
        if "tiktok" in cont.publishers:
            out["tiktok"].update(cont.publishers["tiktok"].token_status())
        return out

    @api.get("/accounts/tiktok/connect")
    def tiktok_connect(request: Request):
        tt = c(request).publishers.get("tiktok")
        if tt is None:
            raise HTTPException(409, "enable publishing.tiktok in config/agent.yaml first")
        try:
            return {"url": tt.authorize_url()}
        except PublishError as e:
            raise HTTPException(409, str(e))

    @api.post("/accounts/tiktok/code")
    async def tiktok_code(body: Code, request: Request):
        tt = c(request).publishers.get("tiktok")
        if tt is None:
            raise HTTPException(409, "TikTok not enabled")
        try:
            return await tt.exchange_code(body.code.strip())
        except PublishError as e:
            raise HTTPException(409, str(e))

    # ---- evals ----------------------------------------------------------
    @api.get("/evals")
    def list_evals(request: Request):
        return c(request).evals.list_runs()

    @api.post("/evals")
    async def start_eval(body: NewEval, request: Request):
        try:
            rid = c(request).evals.start(body.variants, name=body.name, cases=body.cases, limit=body.limit,
                                         types=body.types, auto_judge=body.auto_judge,
                                         concurrency=body.concurrency)
        except (EvalError, ValueError) as e:
            raise HTTPException(422, str(e))
        return {"id": rid}

    @api.get("/evals/{rid}")
    def eval_report(rid: str, request: Request, voter: str | None = None):
        try:
            return c(request).evals.report(rid, voter=voter)
        except EvalError as e:
            raise HTTPException(404, str(e))

    @api.get("/evals/{rid}/pair")
    def eval_pair(rid: str, request: Request):
        try:
            return c(request).evals.next_pair(rid) or {"done": True}
        except EvalError as e:
            raise HTTPException(404, str(e))

    @api.post("/evals/vote")
    def eval_vote(body: Vote, request: Request):
        try:
            c(request).evals.vote(body.token, body.choice)
        except EvalError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @api.get("/gpu")
    def gpu_status(request: Request):
        return c(request).gpu.status()

    @api.post("/gpu/mode")
    async def gpu_mode(body: GpuMode, request: Request):
        """Video side: call with mode=render before rendering (blocks until the GPU is free),
        and mode=think when done. Re-call with render to renew a ttl lease."""
        return await c(request).gpu.set_mode(body.mode, holder=body.holder, ttl_s=body.ttl_s)

    app.include_router(api)

    @app.get("/auth/tiktok/callback")
    async def tiktok_callback(request: Request, code: str = "", state: str = "", error: str = ""):
        """Browser redirect target after TikTok login (protected by the OAuth state value)."""
        tt = c(request).publishers.get("tiktok")
        if error or tt is None or not code:
            return HTMLResponse(f"<p>TikTok connection failed: {html.escape(error or 'not enabled')}</p>", 400)
        try:
            await tt.exchange_code(code, state)
        except PublishError as e:
            return HTMLResponse(f"<p>TikTok connection failed: {html.escape(str(e))}</p>", 400)
        return HTMLResponse("<p>TikTok connected. You can close this tab.</p>")
    return app
