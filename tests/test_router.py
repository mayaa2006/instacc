from agent.container import Container


def build(settings, transport):
    settings.agent.llama_server.manage_processes = False
    return Container.build(settings, transport=transport)


async def test_auto_routes_with_fast_model(settings, transport, fake):
    fake.route_to = "idea"
    c = build(settings, transport)
    tid = c.tasks.create("auto", "a weekly series for sundays")
    await c.tasks.wait_idle()
    t = c.db.task_get(tid)
    assert t["type"] == "idea" and t["status"] == "awaiting_approval"
    assert c.db.trace_get(tid)[0]["stage"] == "route"


async def test_auto_keyword_fallback_on_bad_answer(settings, transport, fake):
    fake.route_to = "nonsense"
    c = build(settings, transport)
    tid = c.tasks.create("auto", "write me a caption with hashtags")
    await c.tasks.wait_idle()
    assert c.db.task_get(tid)["type"] == "caption"


def test_unknown_type_still_rejected(settings, transport):
    import pytest
    c = build(settings, transport)
    with pytest.raises(ValueError):
        c.tasks.create("nope", "x")
