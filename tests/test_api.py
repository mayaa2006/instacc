import time

from fastapi.testclient import TestClient

from agent.api.app import create_app


def wait_status(client, tid, status, headers=None, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        t = client.get(f"/api/tasks/{tid}", headers=headers or {}).json()
        if t["status"] == status:
            return t
        time.sleep(0.05)
    raise AssertionError(f"task stuck in {t['status']}: {t.get('error')}")


def test_end_to_end(settings, transport):
    settings.agent.llama_server.manage_processes = False
    settings.secrets.api_token = "s3cret"
    H = {"Authorization": "Bearer s3cret"}
    with TestClient(create_app(settings, transport=transport, start_bot=False)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/status").status_code == 401
        assert client.get("/api/persona", headers=H).json()["data"]["name"] == "Ava"
        assert "hooks" in client.get("/api/task-types", headers=H).json()

        tid = client.post("/api/tasks", json={"type": "hooks", "prompt": "toast"}, headers=H).json()["id"]
        t = wait_status(client, tid, "awaiting_approval", H)
        assert len(t["result"]["options"]) == 5 and t["trace"]

        assert client.post(f"/api/tasks/{tid}/approve", json={"index": 2}, headers=H).json()["status"] == "approved"
        assert client.post(f"/api/tasks/{tid}/approve", json={"index": 0}, headers=H).status_code == 409

        assert client.post("/api/tasks", json={"type": "nope", "prompt": "x"}, headers=H).status_code == 422

        r = client.put("/api/persona", json={"name": "Ava", "tagline": "new"}, headers=H)
        assert r.json()["version"] == 2

        g = client.post("/api/gpu/mode", json={"mode": "render", "holder": "video", "ttl_s": 60}, headers=H).json()
        assert g["mode"] == "render" and g["holder"] == "video"
        assert client.post("/api/gpu/mode", json={"mode": "think"}, headers=H).json()["mode"] == "think"
        s = client.get("/api/status", headers=H).json()
        assert s["budget"]["spent_usd"] > 0


def test_eval_api(settings, transport):
    settings.agent.llama_server.manage_processes = False
    with TestClient(create_app(settings, transport=transport, start_bot=False)) as client:
        assert "auto" in client.get("/api/task-types").json()
        rid = client.post("/api/evals", json={"variants": ["single:main", "pipeline"], "limit": 2}).json()["id"]
        deadline = time.time() + 10
        while client.get("/api/evals").json()[0]["status"] == "running" and time.time() < deadline:
            time.sleep(0.05)
        rep = client.get(f"/api/evals/{rid}").json()
        assert rep["run"]["status"] == "done" and rep["votes"] == 2
        pair = client.get(f"/api/evals/{rid}/pair").json()
        assert client.post("/api/evals/vote", json={"token": pair["token"], "choice": "tie"}).json()["ok"]
        assert client.post("/api/evals/vote", json={"token": pair["token"], "choice": "tie"}).status_code == 409
        assert client.post("/api/evals", json={"variants": ["bad"]}).status_code == 422


def test_posts_and_render_queue_api(settings, transport, tmp_path):
    settings.agent.llama_server.manage_processes = False
    settings.agent.autopilot.enabled = False
    with TestClient(create_app(settings, transport=transport, start_bot=False)) as client:
        pid = client.post("/api/posts", json={"concept": "toast fail"}).json()["id"]
        deadline = time.time() + 5
        while client.get(f"/api/posts/{pid}").json()["status"] != "script_review" and time.time() < deadline:
            time.sleep(0.05)
        assert client.post(f"/api/posts/{pid}/script/approve").json()["status"] == "render_queued"
        assert [p["id"] for p in client.get("/api/render-queue").json()] == [pid]
        assert client.post(f"/api/posts/{pid}/claim").json()["status"] == "rendering"
        media = tmp_path / "x.mp4"
        media.write_bytes(b"0")
        assert client.post(f"/api/posts/{pid}/rendered", json={"media_path": str(media)}).json()["status"] == "captioning"
        assert client.get(f"/api/posts/{pid}/media").status_code == 200
        assert client.post(f"/api/posts/{pid}/rendered", json={"media_path": "/nope.mp4"}).status_code == 409
        assert client.post("/api/continuity", json={"kind": "outfit", "key": "k", "value": "v"}).status_code == 200
        assert client.post("/api/continuity", json={"kind": "bad", "key": "k", "value": "v"}).status_code == 422
        assert client.get("/api/analytics").json()["calibration"]["n"] == 0
        acc = client.get("/api/accounts").json()
        assert acc["instagram"]["enabled"] is False
        assert client.get("/api/accounts/tiktok/connect").status_code == 409
