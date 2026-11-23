import json
import shutil
from pathlib import Path

import httpx
import pytest

from agent.config import load_settings

ROOT = Path(__file__).resolve().parent.parent


class FakeLLM:
    """Fake OpenAI-compatible backend. Records calls; responses depend on the prompt stage."""

    def __init__(self):
        self.calls: list[dict] = []
        self.judge_scores: list[list[float]] = []  # consumed per judge call; default 8s
        self.cloud_refuses = False
        self.cloud_fails = False
        self.cloud_cost = 0.002
        self.route_to = "idea"
        self.embed_calls: list = []
        self.produce_calls = 0
        self.produce_bad_first = False
        self.reply_text = "haha thank you, the toast thanks you too"

    def handler(self, request: httpx.Request) -> httpx.Response:
        host = f"{request.url.host}:{request.url.port}" if request.url.port else request.url.host
        cloud = "openrouter" in request.url.host
        if request.url.path.endswith("/embeddings"):
            body = json.loads(request.content)
            import hashlib
            data = []
            for i, t in enumerate(body["input"]):
                h = hashlib.sha256(t.encode()).digest()
                data.append({"index": i, "embedding": [b - 128.0 for b in h[:16]]})
            self.embed_calls.append(body["input"])
            return httpx.Response(200, json={"data": data})
        body = json.loads(request.content)
        user = body["messages"][-1]["content"]
        self.calls.append({"host": host, "cloud": cloud, "model": body["model"], "user": user,
                           "system": body["messages"][0]["content"], "body": body,
                           "temperature": body.get("temperature")})
        if cloud and self.cloud_fails:
            return httpx.Response(503, text="upstream down")
        if "Did the model REFUSE" in user:
            content = "YES"
        elif "You are the director" in body["messages"][0]["content"]:
            self.produce_calls += 1
            if self.produce_bad_first and self.produce_calls == 1:
                content = '{"title": "x", "shots": []}'
            else:
                content = json.dumps({"title": "Toast disaster", "concept": "burnt toast", "continuity": {
                    "outfit": "cream knit sweater", "location": "apartment kitchen", "lighting": "morning sun"},
                    "shots": [{"duration_s": 4, "camera": "handheld close-up", "action": "toast pops out black",
                               "dialogue": "growth."},
                              {"duration_s": 3, "camera": "medium, static", "action": "she eats it anyway"}],
                    "audio": {"music_mood": "lofi"}})
        elif "Extract continuity facts" in user:
            content = json.dumps({"facts": [{"kind": "outfit", "key": "signature", "value": "cream knit sweater"},
                                            {"kind": "joke", "key": "toast", "value": "she always burns toast"},
                                            {"kind": "nonsense", "key": "x", "value": "y"}]})
        elif "Classify a follower comment" in user:
            body_text = user.split("<<<")[1].split(">>>")[0].lower()
            if "followers" in body_text:
                content = json.dumps({"category": "spam", "risk": "low", "needs_reply": False})
            elif "?" in body_text:
                content = json.dumps({"category": "question", "risk": "medium", "needs_reply": True})
            else:
                content = json.dumps({"category": "praise", "risk": "low", "needs_reply": True})
        elif "You reply to a comment" in user:
            content = self.reply_text
        elif "Classify this request" in user:
            content = json.dumps({"type": self.route_to, "reason": "test"})
        elif "judging two pieces" in user:
            o1 = user.split("OPTION 1:")[1].split("OPTION 2:")[0]
            o2 = user.split("OPTION 2:")[1]
            # prefers synthesized ("final") content regardless of position
            w = "1" if "final" in o1 and "final" not in o2 else "2" if "final" in o2 and "final" not in o1 else "tie"
            content = json.dumps({"winner": w, "reason": "test"})
        elif cloud and self.cloud_refuses:
            content = "I'm sorry, but I can't help with that request."
        elif "impartial judge" in user:
            n = user.count("\n[")  + (1 if user.split("OPTIONS:")[1].lstrip().startswith("[") else 0)
            scores = self.judge_scores.pop(0) if self.judge_scores else [8.0] * 5
            content = json.dumps({"scores": [{"id": i + 1, "score": s, "notes": "ok"} for i, s in enumerate(scores)],
                                  "feedback": "make it punchier"})
        elif "reviewing options" in user:
            content = "```json\n" + json.dumps({"critiques": [{"id": 1, "strengths": ["a"], "weaknesses": ["b"],
                                                               "issues": [], "steal_this": "x", "score": 6}]}) + "\n```"
        elif "final writer" in user:
            content = json.dumps({"options": [{"text": f"final {i}", "rationale": "fused"} for i in range(5)],
                                  "decision_log": ["kept hook from [1]"]})
        else:
            content = "Sure! " + json.dumps({"options": [{"text": f"{body['model']} idea {i}", "rationale": "r"}
                                                          for i in range(3)]})
        usage = {"prompt_tokens": 100, "completion_tokens": 50}
        if cloud:
            usage["cost"] = self.cloud_cost
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                         "usage": usage})


@pytest.fixture
def fake():
    return FakeLLM()


@pytest.fixture
def transport(fake):
    return httpx.MockTransport(fake.handler)


@pytest.fixture
def root(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "config", tmp_path / "config")
    shutil.copytree(ROOT / "evals", tmp_path / "evals")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    for k in ("INSTACC_API_TOKEN", "TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USERS"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


@pytest.fixture
def settings(root):
    """Cloud profile by default so routing tests have real cloud slots; see local_settings."""
    from agent.config import apply_slot_profile
    s = load_settings(root)
    s.agent.data_dir = str(root / "data")
    s.secrets.api_token = None   # tests opt in to auth explicitly
    apply_slot_profile(s.agent, "cloud")
    return s


@pytest.fixture
def local_settings(root):
    """Settings exactly as shipped (slot_profile: local)."""
    s = load_settings(root)
    s.agent.data_dir = str(root / "data")
    s.secrets.api_token = None
    s.agent.llama_server.manage_processes = False
    return s


class FakePlatforms:
    """Fake Instagram Graph + rupload + TikTok open API."""

    def __init__(self):
        self.calls: list[httpx.Request] = []
        self.ig_status = ["IN_PROGRESS", "FINISHED"]
        self.ig_token_expired = False
        self.ig_bad_metric = None
        self.tt_status = ["PROCESSING_UPLOAD", "PUBLISH_COMPLETE"]
        self.tt_refreshes = 0
        self.replies: list[tuple[str, str]] = []
        self.comments = [
            {"id": "c1", "text": "love this so much", "username": "fan1", "timestamp": "2026-11-20T10:00:00+0000",
             "replies": {"data": [{"id": "c1r", "text": "same!!", "username": "fan2",
                                   "timestamp": "2026-11-20T10:05:00+0000"}]}},
            {"id": "c2", "text": "buy followers cheap", "username": "bot", "timestamp": "2026-11-20T11:00:00+0000"},
            {"id": "c3", "text": "what brush do you use?", "username": "artist", "timestamp": "2026-11-20T12:00:00+0000"},
        ]

    def handler(self, r: httpx.Request) -> httpx.Response:
        self.calls.append(r)
        host, path, q = r.url.host, r.url.path, dict(r.url.params)
        if host in ("graph.facebook.com", "graph.instagram.com"):
            if self.ig_token_expired:
                return httpx.Response(400, json={"error": {"message": "Session has expired", "code": 190}})
            if path.endswith("/media") and r.method == "POST":
                return httpx.Response(200, json={"id": "cont1", "uri": "https://rupload.facebook.com/ig-api-upload/v27.0/cont1"})
            if path.endswith("/cont1"):
                return httpx.Response(200, json={"status_code": self.ig_status.pop(0) if self.ig_status else "FINISHED"})
            if path.endswith("/media_publish"):
                return httpx.Response(200, json={"id": "media9"})
            if path.endswith("/media9") and q.get("fields") == "permalink":
                return httpx.Response(200, json={"permalink": "https://www.instagram.com/reel/abc/"})
            if path.endswith("/media9/comments"):
                return httpx.Response(200, json={"data": self.comments})
            if path.endswith("/replies"):
                self.replies.append((path.split("/")[-2], q.get("message")))
                return httpx.Response(200, json={"id": f"rep{len(self.replies)}"})
            if path.endswith("/insights"):
                metrics = q["metric"].split(",")
                if self.ig_bad_metric and self.ig_bad_metric in metrics:
                    return httpx.Response(400, json={"error": {"message": f"(#100) metric[0] must be one of ... not {self.ig_bad_metric}", "code": 100}})
                vals = {"views": 1000, "reach": 800, "likes": 90, "comments": 5, "shares": 4, "saved": 1}
                return httpx.Response(200, json={"data": [{"name": m, "values": [{"value": vals.get(m, 1)}]} for m in metrics]})
        if host == "rupload.facebook.com":
            return httpx.Response(200, json={"success": True})
        if host == "open.tiktokapis.com":
            if path == "/v2/oauth/token/":
                self.tt_refreshes += 1
                return httpx.Response(200, json={"access_token": f"at{self.tt_refreshes}", "refresh_token": f"rt{self.tt_refreshes}",
                                                 "expires_in": 7200, "refresh_expires_in": 15552000, "open_id": "u1",
                                                 "scope": "video.publish"})
            if path.endswith("creator_info/query/"):
                return httpx.Response(200, json={"data": {"privacy_level_options": ["SELF_ONLY", "FOLLOWER_OF_CREATOR"]},
                                                 "error": {"code": "ok"}})
            if path.endswith("/video/init/"):
                return httpx.Response(200, json={"data": {"publish_id": "pub1", "upload_url": "https://upload.tiktok.example/u1"},
                                                 "error": {"code": "ok"}})
            if path.endswith("/status/fetch/"):
                st = self.tt_status.pop(0) if self.tt_status else "PUBLISH_COMPLETE"
                data = {"status": st}
                if st == "PUBLISH_COMPLETE":
                    data["publicaly_available_post_id"] = [7123]
                return httpx.Response(200, json={"data": data, "error": {"code": "ok"}})
            if path == "/v2/video/query/":
                return httpx.Response(200, json={"data": {"videos": [{"id": 7123, "view_count": 500, "like_count": 50,
                                                                       "comment_count": 2, "share_count": 3}]},
                                                 "error": {"code": "ok"}})
        if host == "upload.tiktok.example":
            return httpx.Response(201)
        return httpx.Response(404, json={"error": {"message": f"unhandled {r.method} {r.url}"}})


@pytest.fixture
def platforms():
    return FakePlatforms()


async def _nosleep(_s):
    return None


@pytest.fixture
def publishing(settings, platforms, monkeypatch):
    """Settings with both platforms enabled and fake credentials."""
    pub = settings.agent.publishing
    pub.instagram.enabled = True
    pub.tiktok.enabled = True
    pub.tiktok.redirect_uri = "https://example.com/cb"
    pub.tiktok.mode = "direct"
    monkeypatch.setenv("IG_USER_ID", "1789")
    monkeypatch.setenv("IG_ACCESS_TOKEN", "igtok")
    monkeypatch.setenv("TIKTOK_CLIENT_KEY", "ck")
    monkeypatch.setenv("TIKTOK_CLIENT_SECRET", "cs")
    settings.agent.llama_server.manage_processes = False
    settings.agent.autopilot.enabled = False
    return settings
