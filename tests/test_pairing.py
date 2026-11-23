import time

import pytest
from fastapi.testclient import TestClient

from agent.api.app import create_app
from agent.config import ensure_api_token, load_settings
from agent.net import _kind, phone_addresses
from agent.pairing import Pairing, PairingError, qr_svg, qr_terminal


def test_codes_are_single_use_and_expire(monkeypatch):
    p = Pairing("secret", port=8123)
    code, exp = p.new_code()
    assert exp > time.time() + 500
    assert p.claim(code) == "secret"
    with pytest.raises(PairingError):
        p.claim(code)                       # single use
    code2, _ = p.new_code()
    p._codes[code2] = time.time() - 1       # expired
    with pytest.raises(PairingError):
        p.claim(code2)


def test_brute_force_is_throttled():
    p = Pairing("secret")
    for _ in range(10):
        with pytest.raises(PairingError):
            p.claim("guess")
    good, _ = p.new_code()
    with pytest.raises(PairingError, match="Too many"):
        p.claim(good)


def test_links_use_fragment(monkeypatch):
    monkeypatch.setattr("agent.net.phone_addresses",
                        lambda: [{"ip": "192.168.1.23", "kind": "wifi"}, {"ip": "100.101.102.103", "kind": "tailscale"}])
    p = Pairing("s", port=8000, scheme="https")
    links = p.links("abc")
    assert links[0] == {"label": "Home Wi-Fi", "url": "https://192.168.1.23:8000/#pair=abc"}
    assert links[1]["label"].startswith("Tailscale")
    assert p.pc_link("x") == "https://127.0.0.1:8000/#pair=x"


def test_address_classification():
    assert _kind("192.168.1.5") == "wifi" and _kind("10.0.0.7") == "wifi" and _kind("172.20.1.1") == "wifi"
    assert _kind("100.88.1.2") == "tailscale"
    assert _kind("127.0.0.1") is None and _kind("169.254.3.3") is None and _kind("8.8.8.8") is None
    assert all(a["kind"] in ("wifi", "tailscale") for a in phone_addresses())


def test_qr_outputs():
    svg = qr_svg("http://192.168.1.23:8000/#pair=abc")
    assert svg.startswith(b"<?xml") and b"<svg" in svg
    assert "█" in qr_terminal("http://x")


def test_password_auto_created(root, monkeypatch):
    monkeypatch.delenv("INSTACC_API_TOKEN", raising=False)
    (root / ".env").write_text("OPENROUTER_API_KEY=\nINSTACC_API_TOKEN=\n", encoding="utf-8")
    s = load_settings(root)
    tok = s.secrets.api_token
    assert tok and len(tok) >= 30
    env = (root / ".env").read_text()
    assert env.count("INSTACC_API_TOKEN=") == 1 and f"INSTACC_API_TOKEN={tok}" in env and "OPENROUTER_API_KEY=" in env
    monkeypatch.delenv("INSTACC_API_TOKEN", raising=False)
    assert load_settings(root).secrets.api_token == tok     # stable across restarts
    assert ensure_api_token(root / "new.env") != tok


def test_pair_api_flow(settings, transport, monkeypatch):
    monkeypatch.setattr("agent.net.phone_addresses", lambda: [{"ip": "192.168.1.23", "kind": "wifi"}])
    settings.agent.llama_server.manage_processes = False
    settings.agent.autopilot.enabled = False
    settings.secrets.api_token = "s3cret"
    H = {"Authorization": "Bearer s3cret"}
    with TestClient(create_app(settings, transport=transport, start_bot=False, port=8000)) as client:
        assert client.post("/api/pair/new").status_code == 401          # only a signed-in device can make codes
        p = client.post("/api/pair/new", headers=H).json()
        assert p["links"][0]["url"] == f"http://192.168.1.23:8000/#pair={p['code']}"
        qr = client.get("/api/pair/qr.svg", params={"data": p["links"][0]["url"], "token": "s3cret"})
        assert qr.status_code == 200 and qr.headers["content-type"].startswith("image/svg")
        assert client.get("/api/pair/qr.svg", params={"data": "javascript:alert(1)", "token": "s3cret"}).status_code == 422
        r = client.post("/pair/claim", json={"code": p["code"]})          # no auth header needed
        assert r.status_code == 200 and r.json()["token"] == "s3cret"
        assert client.post("/pair/claim", json={"code": p["code"]}).status_code == 403
        assert client.get("/api/status", headers={"Authorization": f"Bearer {r.json()['token']}"}).status_code == 200
