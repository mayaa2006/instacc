from agent.container import Container


def build(settings, transport):
    settings.agent.llama_server.manage_processes = False
    return Container.build(settings, transport=transport)


MSG = [{"role": "user", "content": "write a hook"}]


async def test_private_never_goes_to_cloud(settings, transport, fake):
    c = build(settings, transport)
    r = await c.llm.chat("judge", MSG, sensitivity="private")
    assert r.kind == "local" and r.slot == "main" and r.rerouted == "private->local"
    assert not any(call["cloud"] for call in fake.calls)


async def test_public_uses_cloud_and_tracks_cost(settings, transport, fake):
    c = build(settings, transport)
    r = await c.llm.chat("judge", MSG, sensitivity="public")
    assert r.kind == "cloud" and r.rerouted is None
    assert abs(c.db.cloud_spend_this_month() - 0.002) < 1e-9


async def test_refusal_falls_back_to_local(settings, transport, fake):
    fake.cloud_refuses = True
    c = build(settings, transport)
    r = await c.llm.chat("alt", MSG, sensitivity="public")
    assert r.kind == "local" and r.rerouted == "refusal->local"


async def test_cloud_error_falls_back(settings, transport, fake):
    fake.cloud_fails = True
    c = build(settings, transport)
    r = await c.llm.chat("alt", MSG, sensitivity="public")
    assert r.kind == "local" and r.rerouted.startswith("cloud_error")


async def test_budget_exhausted_routes_local(settings, transport, fake):
    settings.agent.budget.monthly_usd = 0.001
    fake.cloud_cost = 0.01
    c = build(settings, transport)
    first = await c.llm.chat("alt", MSG, sensitivity="public")
    second = await c.llm.chat("alt", MSG, sensitivity="public")
    assert first.kind == "cloud" and second.kind == "local" and second.rerouted == "budget->local"


async def test_thinking_disabled_on_local_only(settings, transport, fake):
    c = build(settings, transport)
    await c.llm.chat("main", MSG)
    await c.llm.chat("judge", MSG, sensitivity="public")
    local, cloud = fake.calls[-2]["body"], fake.calls[-1]["body"]
    assert local["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in cloud


async def test_unclosed_think_and_truncation(settings):
    import httpx
    import pytest
    from agent.models.client import LLMError
    replies = iter([
        {"content": "<think>still reasoning about the milk...", "finish_reason": "length"},
        {"content": "<think>ok</think>  Real answer.", "finish_reason": "stop"},
    ])

    def handler(r):
        m = next(replies)
        return httpx.Response(200, json={"choices": [{"message": {"content": m["content"]},
                                                      "finish_reason": m["finish_reason"]}], "usage": {}})
    c = build(settings, httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="ran out of tokens"):
        await c.llm.chat("main", MSG)
    assert (await c.llm.chat("main", MSG)).text == "Real answer."
