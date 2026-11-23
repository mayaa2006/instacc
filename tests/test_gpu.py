import asyncio

from agent.db import DB
from agent.scheduler.gpu import GpuScheduler
from agent.scheduler.servers import ServerManager


async def test_private_calls_wait_during_render(settings, tmp_path):
    settings.agent.llama_server.manage_processes = True  # models don't exist -> nothing actually starts
    gpu = GpuScheduler(DB(tmp_path / "g.db"), ServerManager(settings), comfyui_url=None)
    await gpu.start()
    await gpu.set_mode("render", holder="video", ttl_s=60)

    got = asyncio.Event()

    async def call_main():
        async with gpu.lease("main"):
            got.set()

    t = asyncio.create_task(call_main())
    await asyncio.sleep(0.05)
    assert not got.is_set() and gpu.status()["calls_waiting_for_think_mode"] == 1

    async with gpu.lease("embed"):  # embed has a render profile -> not blocked
        pass

    await gpu.set_mode("think")
    await asyncio.wait_for(t, 1)
    assert got.is_set()


async def test_render_lease_expires(settings, tmp_path):
    settings.agent.llama_server.manage_processes = False
    gpu = GpuScheduler(DB(tmp_path / "g.db"), ServerManager(settings), comfyui_url=None)
    await gpu.start()
    await gpu.set_mode("render", holder="video", ttl_s=0.05)
    await asyncio.sleep(0.2)
    assert gpu.mode == "think" and gpu.holder is None


async def test_switch_waits_for_inflight(settings, tmp_path):
    settings.agent.llama_server.manage_processes = False
    gpu = GpuScheduler(DB(tmp_path / "g.db"), ServerManager(settings), comfyui_url=None)
    await gpu.start()
    order = []

    async def long_call():
        async with gpu.lease("main"):
            await asyncio.sleep(0.1)
            order.append("call_done")

    t = asyncio.create_task(long_call())
    await asyncio.sleep(0.01)
    await gpu.set_mode("render", holder="video")
    order.append("switched")
    await t
    assert order == ["call_done", "switched"]


def test_mode_persists(settings, tmp_path):
    db = DB(tmp_path / "g.db")
    db.kv_set("gpu", {"mode": "render", "holder": "video", "since": 1, "expires_at": None})
    assert GpuScheduler(db, ServerManager(settings)).mode == "render"


def test_binary_resolves_relative(settings):
    sm = ServerManager(settings)
    assert sm.binary().replace("\\", "/").endswith("llm/llama.cpp/llama-server.exe")
    cmd = sm.command("main", "think")
    assert "-ngl" in cmd and "--port" in cmd
