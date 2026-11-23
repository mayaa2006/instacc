"""Instagram Reels publishing, comments and insights via the official Instagram Platform API.

Verified against Meta docs (2026-11, API v27.0):
- POST /<IG_ID>/media  media_type=REELS (+ upload_type=resumable | video_url), caption, is_ai_generated
- Resumable upload (Facebook Login only): POST https://rupload.facebook.com/ig-api-upload/<ver>/<container>
  headers Authorization: OAuth <token>, offset: 0, file_size: <bytes>
- GET /<container>?fields=status_code -> IN_PROGRESS | FINISHED | ERROR | EXPIRED | PUBLISHED
- POST /<IG_ID>/media_publish creation_id=<container>
- GET /<media>/comments ; POST /<comment>/replies message=...
- GET /<media>/insights?metric=...
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import httpx

from agent.config import InstagramCfg
from agent.publish.base import AuthError, PublishError, PublishResult

log = logging.getLogger(__name__)


class InstagramClient:
    platform = "instagram"

    def __init__(self, cfg: InstagramCfg, ai_label: bool = True, transport: httpx.AsyncBaseTransport | None = None,
                 sleep=asyncio.sleep):
        self.cfg = cfg
        self.ai_label = ai_label
        self.http = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(120, connect=15))
        self._sleep = sleep

    async def aclose(self) -> None:
        await self.http.aclose()

    @property
    def base(self) -> str:
        host = "graph.facebook.com" if self.cfg.login == "facebook" else "graph.instagram.com"
        return f"https://{host}/{self.cfg.api_version}"

    def _creds(self) -> tuple[str, str]:
        uid, tok = os.environ.get(self.cfg.user_id_env), os.environ.get(self.cfg.token_env)
        if not uid or not tok:
            raise AuthError(f"set {self.cfg.user_id_env} and {self.cfg.token_env} in .env")
        return uid, tok

    def configured(self) -> bool:
        try:
            self._creds()
            return True
        except AuthError:
            return False

    async def _req(self, method: str, url: str, **kw) -> dict:
        _, tok = self._creds()
        params = kw.pop("params", {}) or {}
        params.setdefault("access_token", tok)
        r = await self.http.request(method, url, params=params, **kw)
        try:
            data = r.json()
        except ValueError:
            data = {"raw": r.text[:300]}
        if r.status_code >= 400 or "error" in data:
            err = data.get("error", {}) if isinstance(data, dict) else {}
            msg = f"Instagram {r.status_code}: {err.get('message') or data}"
            if err.get("code") == 190 or r.status_code == 401:
                raise AuthError(msg + " (token expired or revoked; generate a new one)")
            raise PublishError(msg)
        return data

    async def publish_reel(self, video: Path, caption: str) -> PublishResult:
        uid, tok = self._creds()
        video = Path(video)
        if not video.exists():
            raise PublishError(f"video not found: {video}")
        params = {"media_type": "REELS", "caption": caption}
        if self.cfg.share_to_feed:
            params["share_to_feed"] = "true"
        if self.ai_label:
            params["is_ai_generated"] = "true"

        if self.cfg.login == "facebook":
            params["upload_type"] = "resumable"
            created = await self._req("POST", f"{self.base}/{uid}/media", params=params)
            cid = created["id"]
            upload_uri = created.get("uri") or f"https://rupload.facebook.com/ig-api-upload/{self.cfg.api_version}/{cid}"
            size = video.stat().st_size
            r = await self.http.post(upload_uri, content=video.read_bytes(), timeout=600, headers={
                "Authorization": f"OAuth {tok}", "offset": "0", "file_size": str(size)})
            if r.status_code >= 400:
                raise PublishError(f"Instagram upload {r.status_code}: {r.text[:300]}")
        else:
            if not self.cfg.public_media_base_url:
                raise PublishError("login=instagram needs publishing.instagram.public_media_base_url "
                                   "(videos must be at a public URL), or switch to login=facebook")
            params["video_url"] = self.cfg.public_media_base_url.rstrip("/") + "/" + video.name
            cid = (await self._req("POST", f"{self.base}/{uid}/media", params=params))["id"]

        await self._wait_container(cid)
        media_id = (await self._req("POST", f"{self.base}/{uid}/media_publish", params={"creation_id": cid}))["id"]
        url = None
        try:
            url = (await self._req("GET", f"{self.base}/{media_id}", params={"fields": "permalink"})).get("permalink")
        except PublishError as e:
            log.info("couldn't fetch permalink: %s", e)
        return PublishResult(self.platform, media_id, url, "published", {"container_id": cid})

    async def _wait_container(self, cid: str) -> None:
        waited = 0.0
        while True:
            st = (await self._req("GET", f"{self.base}/{cid}", params={"fields": "status_code,status"}))
            code = st.get("status_code")
            if code in ("FINISHED", "PUBLISHED"):
                return
            if code in ("ERROR", "EXPIRED"):
                raise PublishError(f"Instagram container {cid} {code}: {st.get('status')}")
            if waited >= self.cfg.poll_max_s:
                raise PublishError(f"Instagram container {cid} still {code} after {waited:.0f}s")
            await self._sleep(self.cfg.poll_interval_s)
            waited += self.cfg.poll_interval_s

    async def comments(self, media_id: str) -> list[dict]:
        """Top-level comments + replies, flattened. Each: id, text, username, timestamp, parent_id."""
        out, url = [], f"{self.base}/{media_id}/comments"
        params = {"fields": "id,text,username,timestamp,replies{id,text,username,timestamp}", "limit": "50"}
        for _ in range(10):  # pages
            data = await self._req("GET", url, params=params)
            for c in data.get("data", []):
                out.append({**{k: c.get(k) for k in ("id", "text", "username", "timestamp")}, "parent_id": None})
                for rep in (c.get("replies") or {}).get("data", []):
                    out.append({**{k: rep.get(k) for k in ("id", "text", "username", "timestamp")},
                                "parent_id": c["id"]})
            nxt = (data.get("paging") or {}).get("next")
            if not nxt:
                break
            url, params = nxt, {}
        return out

    async def reply(self, comment_id: str, message: str) -> str:
        return (await self._req("POST", f"{self.base}/{comment_id}/replies", params={"message": message}))["id"]

    async def insights(self, media_id: str) -> dict:
        """Fetch configured metrics; drops metrics the API rejects and retries once."""
        metrics = list(self.cfg.insights_metrics)
        for _ in range(3):
            try:
                data = await self._req("GET", f"{self.base}/{media_id}/insights", params={"metric": ",".join(metrics)})
                return {m["name"]: (m.get("values") or [{}])[0].get("value", m.get("total_value", {}).get("value"))
                        for m in data.get("data", [])}
            except PublishError as e:
                bad = [m for m in metrics if m in str(e)]
                if not bad or isinstance(e, AuthError):
                    raise
                log.warning("Instagram rejected metrics %s; retrying without them", bad)
                metrics = [m for m in metrics if m not in bad]
        return {}
