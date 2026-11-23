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
