import time

import httpx
import pytest

from agent.db import DB
from agent.publish.base import AuthError
from agent.publish.instagram import InstagramClient
from agent.publish.tiktok import SINGLE_MAX, TikTokClient
from tests.conftest import _nosleep


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00" * 2048)
    return p


async def test_instagram_resumable_reel(publishing, platforms, video):
    ig = InstagramClient(publishing.agent.publishing.instagram, True, httpx.MockTransport(platforms.handler), _nosleep)
    r = await ig.publish_reel(video, "caption #toast")
    assert r.external_id == "media9" and r.url.startswith("https://www.instagram.com/reel/")
    create = platforms.calls[0]
    q = dict(create.url.params)
    assert create.url.host == "graph.facebook.com" and "/v27.0/1789/media" in create.url.path
    assert q["media_type"] == "REELS" and q["upload_type"] == "resumable" and q["is_ai_generated"] == "true"
    up = next(c for c in platforms.calls if c.url.host == "rupload.facebook.com")
    assert up.headers["authorization"] == "OAuth igtok" and up.headers["offset"] == "0"
    assert up.headers["file_size"] == "2048" and up.content == video.read_bytes()
    assert any(c.url.path.endswith("/media_publish") and dict(c.url.params)["creation_id"] == "cont1"
               for c in platforms.calls)


async def test_instagram_login_needs_public_url(publishing, platforms, video):
    cfg = publishing.agent.publishing.instagram
    cfg.login = "instagram"
    ig = InstagramClient(cfg, True, httpx.MockTransport(platforms.handler), _nosleep)
    with pytest.raises(Exception, match="public"):
        await ig.publish_reel(video, "x")
    cfg.public_media_base_url = "https://media.example.com/v/"
    await ig.publish_reel(video, "x")
    q = dict(platforms.calls[0].url.params)
    assert platforms.calls[0].url.host == "graph.instagram.com" and q["video_url"] == "https://media.example.com/v/clip.mp4"


async def test_instagram_expired_token_is_auth_error(publishing, platforms, video):
    platforms.ig_token_expired = True
    ig = InstagramClient(publishing.agent.publishing.instagram, True, httpx.MockTransport(platforms.handler), _nosleep)
    with pytest.raises(AuthError):
        await ig.publish_reel(video, "x")


async def test_instagram_container_error(publishing, platforms, video):
    platforms.ig_status = ["ERROR"]
    ig = InstagramClient(publishing.agent.publishing.instagram, True, httpx.MockTransport(platforms.handler), _nosleep)
    with pytest.raises(Exception, match="ERROR"):
        await ig.publish_reel(video, "x")


async def test_instagram_insights_drops_rejected_metric(publishing, platforms):
    platforms.ig_bad_metric = "total_interactions"
    ig = InstagramClient(publishing.agent.publishing.instagram, True, httpx.MockTransport(platforms.handler), _nosleep)
    m = await ig.insights("media9")
    assert m["views"] == 1000 and "total_interactions" not in m


async def test_tiktok_oauth_refresh_rotates(publishing, platforms, tmp_path):
    db = DB(tmp_path / "t.db")
    tt = TikTokClient(publishing.agent.publishing.tiktok, db, True, httpx.MockTransport(platforms.handler), _nosleep)
    url = tt.authorize_url()
    assert "client_key=ck" in url and "video.publish" in url and "redirect_uri=https%3A%2F%2Fexample.com%2Fcb" in url
    state = db.kv_get("tiktok_oauth_state")
    with pytest.raises(AuthError):
        await tt.exchange_code("code1", "wrong-state")
    await tt.exchange_code("code1", state)
    assert await tt.access_token() == "at1"
    tok = db.kv_get("tiktok_tokens")
    tok["expires_at"] = time.time() + 10   # about to expire
    db.kv_set("tiktok_tokens", tok)
    assert await tt.access_token() == "at2"
    assert db.kv_get("tiktok_tokens")["refresh_token"] == "rt2"   # rotated token stored
    form = dict(x.split("=") for x in platforms.calls[-1].content.decode().split("&"))
    assert form["grant_type"] == "refresh_token" and form["refresh_token"] == "rt1"


async def test_tiktok_direct_post(publishing, platforms, tmp_path, video):
    db = DB(tmp_path / "t.db")
    tt = TikTokClient(publishing.agent.publishing.tiktok, db, True, httpx.MockTransport(platforms.handler), _nosleep)
    db.kv_set("tiktok_tokens", {"access_token": "at", "refresh_token": "rt", "expires_at": time.time() + 7000,
                                "refresh_expires_at": time.time() + 1e6})
    r = await tt.publish_video(video, "my caption")
    assert r.external_id == "7123" and r.state == "published"
    init = next(c for c in platforms.calls if c.url.path.endswith("/post/publish/video/init/"))
    import json
    body = json.loads(init.content)
    assert body["post_info"]["is_aigc"] is True and body["post_info"]["privacy_level"] == "SELF_ONLY"
    assert body["source_info"] == {"source": "FILE_UPLOAD", "video_size": 2048, "chunk_size": 2048, "total_chunk_count": 1}
    put = next(c for c in platforms.calls if c.url.host == "upload.tiktok.example")
    assert put.headers["content-range"] == "bytes 0-2047/2048" and put.headers["content-type"] == "video/mp4"
    assert init.headers["authorization"] == "Bearer at"


async def test_tiktok_inbox_mode(publishing, platforms, tmp_path, video):
    cfg = publishing.agent.publishing.tiktok
    cfg.mode = "inbox"
    platforms.tt_status = ["SEND_TO_USER_INBOX"]
    db = DB(tmp_path / "t.db")
    db.kv_set("tiktok_tokens", {"access_token": "at", "refresh_token": "rt", "expires_at": time.time() + 7000,
                                "refresh_expires_at": time.time() + 1e6})
    tt = TikTokClient(cfg, db, True, httpx.MockTransport(platforms.handler), _nosleep)
    r = await tt.publish_video(video, "x")
    assert r.state == "in_drafts"
    assert any(c.url.path == "/v2/post/publish/inbox/video/init/" for c in platforms.calls)
    assert not any("creator_info" in c.url.path for c in platforms.calls)


def test_tiktok_chunk_plan():
    assert TikTokClient.chunk_plan(1000) == (1000, 1)
    size = SINGLE_MAX + 5 * 1024 * 1024 + 7
    chunk, count = TikTokClient.chunk_plan(size)
    assert chunk == 10 * 1024 * 1024 and count == size // chunk and size - chunk * (count - 1) < 2 * chunk
