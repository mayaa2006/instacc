"""TikTok Content Posting API (inbox drafts or direct post) + Display API stats.

Verified against TikTok docs (2026-11):
- POST /v2/post/publish/creator_info/query/           (video.publish)
- POST /v2/post/publish/video/init/  post_info{title, privacy_level, disable_*, is_aigc}, source_info{FILE_UPLOAD...}
- POST /v2/post/publish/inbox/video/init/  source_info only (video.upload) -> lands in the creator's drafts
- PUT upload_url  Content-Type video/mp4, Content-Length, Content-Range: bytes a-b/total
- POST /v2/post/publish/status/fetch/ {publish_id}: PROCESSING_UPLOAD | PROCESSING_DOWNLOAD |
  SEND_TO_USER_INBOX | PUBLISH_COMPLETE | FAILED; public id in data.publicaly_available_post_id (sic)
- POST /v2/oauth/token/ grant_type=refresh_token: access token ~2h, refresh token 180d and ROTATES
- POST /v2/video/query/?fields=... {"filters": {"video_ids": [...]}}   (video.list)
Chunking (TikTok media transfer guide; not re-verified): chunks 5-64 MB, total_chunk_count =
floor(size / chunk_size), the last chunk absorbs the remainder; files <= 64 MB go as one chunk.
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

from agent.config import TikTokCfg
from agent.db import DB
from agent.publish.base import AuthError, PublishError, PublishResult

log = logging.getLogger(__name__)
API = "https://open.tiktokapis.com"
CHUNK = 10 * 1024 * 1024
SINGLE_MAX = 64 * 1024 * 1024
TOKEN_KEY = "tiktok_tokens"


class TikTokClient:
    platform = "tiktok"

    def __init__(self, cfg: TikTokCfg, db: DB, ai_label: bool = True,
                 transport: httpx.AsyncBaseTransport | None = None, sleep=asyncio.sleep):
        self.cfg = cfg
        self.db = db
        self.ai_label = ai_label
        self.http = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(120, connect=15))
        self._sleep = sleep
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self.http.aclose()

    # --------------------------------------------------------------- oauth
    def _app(self) -> tuple[str, str]:
        k, s = os.environ.get(self.cfg.client_key_env), os.environ.get(self.cfg.client_secret_env)
        if not k or not s:
            raise AuthError(f"set {self.cfg.client_key_env} and {self.cfg.client_secret_env} in .env")
        return k, s

    def configured(self) -> bool:
        return bool(self.db.kv_get(TOKEN_KEY))

    def authorize_url(self) -> str:
        key, _ = self._app()
        if not self.cfg.redirect_uri:
            raise AuthError("set publishing.tiktok.redirect_uri (must match your TikTok app settings)")
        state = secrets.token_urlsafe(16)
        self.db.kv_set("tiktok_oauth_state", state)
        return self.cfg.authorize_url + "?" + urlencode({
            "client_key": key, "scope": ",".join(self.cfg.scopes), "response_type": "code",
            "redirect_uri": self.cfg.redirect_uri, "state": state,
        })

    async def exchange_code(self, code: str, state: str | None = None) -> dict:
        expected = self.db.kv_get("tiktok_oauth_state")
        if state is not None and state != expected:
            raise AuthError("OAuth state mismatch; start the TikTok connection again")
        key, sec = self._app()
        tok = await self._token_request({
            "client_key": key, "client_secret": sec, "code": code, "grant_type": "authorization_code",
            "redirect_uri": self.cfg.redirect_uri,
        })
        self.db.kv_set("tiktok_oauth_state", None)
        return {"open_id": tok.get("open_id"), "scope": tok.get("scope")}

    async def _token_request(self, form: dict) -> dict:
        r = await self.http.post(f"{API}/v2/oauth/token/", data=form,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
        data = r.json() if r.content else {}
        if r.status_code >= 400 or "access_token" not in data:
            raise AuthError(f"TikTok token error: {data.get('error_description') or data.get('error') or r.text[:200]}")
        now = time.time()
        stored = {
            "access_token": data["access_token"], "refresh_token": data.get("refresh_token"),
            "expires_at": now + float(data.get("expires_in", 7200)),
            "refresh_expires_at": now + float(data.get("refresh_expires_in", 180 * 86400)),
            "open_id": data.get("open_id"), "scope": data.get("scope"),
        }
        self.db.kv_set(TOKEN_KEY, stored)  # refresh tokens rotate: always store the new one
        return stored

    async def access_token(self) -> str:
        async with self._lock:
            tok = self.db.kv_get(TOKEN_KEY)
            if not tok:
                raise AuthError("TikTok not connected: open Accounts in the dashboard")
            if tok["expires_at"] - time.time() > 300:
                return tok["access_token"]
            if not tok.get("refresh_token") or tok.get("refresh_expires_at", 0) < time.time():
                raise AuthError("TikTok refresh token expired: reconnect TikTok in the dashboard")
            key, sec = self._app()
            new = await self._token_request({"client_key": key, "client_secret": sec,
                                             "grant_type": "refresh_token", "refresh_token": tok["refresh_token"]})
            return new["access_token"]

    def token_status(self) -> dict:
        tok = self.db.kv_get(TOKEN_KEY)
        if not tok:
            return {"connected": False}
        return {"connected": True, "open_id": tok.get("open_id"), "scope": tok.get("scope"),
                "refresh_expires_in_days": round((tok.get("refresh_expires_at", 0) - time.time()) / 86400, 1)}

    # ------------------------------------------------------------ requests
    async def _post(self, path: str, body: dict, params: dict | None = None) -> dict:
        token = await self.access_token()
        r = await self.http.post(f"{API}{path}", json=body, params=params, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"})
        data = r.json() if r.content else {}
        err = data.get("error") or {}
        if r.status_code == 401 or err.get("code") in ("access_token_invalid", "scope_not_authorized"):
            raise AuthError(f"TikTok auth: {err.get('message') or r.text[:200]}")
        if r.status_code >= 400 or (err.get("code") not in (None, "", "ok")):
            raise PublishError(f"TikTok {path} {r.status_code}: {err.get('code')} {err.get('message')}")
        return data.get("data") or {}

    @staticmethod
    def chunk_plan(size: int) -> tuple[int, int]:
        if size <= SINGLE_MAX:
            return size, 1
        return CHUNK, size // CHUNK

    async def _upload(self, upload_url: str, video: Path) -> None:
        size = video.stat().st_size
        chunk, count = self.chunk_plan(size)
        with open(video, "rb") as f:
            for i in range(count):
                start = i * chunk
                end = size - 1 if i == count - 1 else start + chunk - 1
                f.seek(start)
                part = f.read(end - start + 1)
                r = await self.http.put(upload_url, content=part, timeout=600, headers={
                    "Content-Type": "video/mp4", "Content-Length": str(len(part)),
                    "Content-Range": f"bytes {start}-{end}/{size}"})
                if r.status_code not in (200, 201, 206):
                    raise PublishError(f"TikTok upload chunk {i + 1}/{count} failed: {r.status_code} {r.text[:200]}")

    async def publish_video(self, video: Path, caption: str) -> PublishResult:
        video = Path(video)
        if not video.exists():
            raise PublishError(f"video not found: {video}")
        size = video.stat().st_size
        chunk, count = self.chunk_plan(size)
        source = {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": chunk, "total_chunk_count": count}
        if self.cfg.mode == "inbox":
            init = await self._post("/v2/post/publish/inbox/video/init/", {"source_info": source})
        else:
            info = await self._post("/v2/post/publish/creator_info/query/", {})
            options = info.get("privacy_level_options") or []
            privacy = self.cfg.privacy_level if self.cfg.privacy_level in options else (
                "SELF_ONLY" if "SELF_ONLY" in options else (options[0] if options else self.cfg.privacy_level))
            post_info = {"title": caption[:2200], "privacy_level": privacy, "disable_comment": False,
                         "disable_duet": False, "disable_stitch": False, "video_cover_timestamp_ms": 1000}
            if self.ai_label:
                post_info["is_aigc"] = True
            init = await self._post("/v2/post/publish/video/init/", {"post_info": post_info, "source_info": source})
        publish_id, upload_url = init["publish_id"], init["upload_url"]
        await self._upload(upload_url, video)
        return await self._wait(publish_id)

    async def _wait(self, publish_id: str) -> PublishResult:
        waited = 0.0
        while True:
            st = await self._post("/v2/post/publish/status/fetch/", {"publish_id": publish_id})
            status = st.get("status")
            if status == "FAILED":
                raise PublishError(f"TikTok publish failed: {st.get('fail_reason')}")
            if status == "SEND_TO_USER_INBOX":
                return PublishResult(self.platform, None, None, "in_drafts", {"publish_id": publish_id})
            if status == "PUBLISH_COMPLETE":
                ids = st.get("publicaly_available_post_id") or st.get("publicly_available_post_id") or []
                pid = str(ids[0]) if ids else None
                state = "published" if pid else ("private" if self.cfg.privacy_level == "SELF_ONLY" else "processing")
                return PublishResult(self.platform, pid, f"https://www.tiktok.com/video/{pid}" if pid else None,
                                     state, {"publish_id": publish_id})
            if waited >= self.cfg.poll_max_s:
                return PublishResult(self.platform, None, None, "processing", {"publish_id": publish_id})
            await self._sleep(self.cfg.poll_interval_s)
            waited += self.cfg.poll_interval_s

    async def video_stats(self, video_ids: list[str]) -> dict[str, dict]:
        fields = "id,view_count,like_count,comment_count,share_count"
        out = {}
        for i in range(0, len(video_ids), 20):
            data = await self._post("/v2/video/query/", {"filters": {"video_ids": video_ids[i:i + 20]}},
                                    params={"fields": fields})
            for v in data.get("videos", []):
                out[str(v["id"])] = {k: v.get(k) for k in ("view_count", "like_count", "comment_count", "share_count")}
        return out
