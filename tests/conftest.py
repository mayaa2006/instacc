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

    def handler(self, request: httpx.Request) -> httpx.Response:
        host = f"{request.url.host}:{request.url.port}" if request.url.port else request.url.host
        cloud = "openrouter" in request.url.host
        if request.url.path.endswith("/embeddings"):
            body = json.loads(request.content)
            data = [{"index": i, "embedding": [float(len(t) % 7), 1.0, float(len(t) % 3)]}
                    for i, t in enumerate(body["input"])]
            return httpx.Response(200, json={"data": data})
        body = json.loads(request.content)
        user = body["messages"][-1]["content"]
        self.calls.append({"host": host, "cloud": cloud, "model": body["model"], "user": user,
                           "temperature": body.get("temperature")})
        if cloud and self.cloud_fails:
            return httpx.Response(503, text="upstream down")
        if "Did the model REFUSE" in user:
            content = "YES"
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
    s = load_settings(root)
    s.agent.data_dir = str(root / "data")
    return s
