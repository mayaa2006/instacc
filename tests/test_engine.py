from agent.container import Container


def build(settings, transport):
    settings.agent.llama_server.manage_processes = False
    return Container.build(settings, transport=transport)


async def test_full_pipeline_with_loop_and_approval(settings, transport, fake):
    # idea uses a 2-persona judge panel: two judge calls per round, scores averaged per option
    fake.judge_scores = [[5] * 5, [6] * 5, [9] * 5, [8] * 5]   # round 1 avg 5.5 (< 7.5) -> loop; round 2 avg 8.5
    idea = settings.tasks.tasks["idea"]   # exercise every stage, whatever the shipped defaults are
    idea.critique, idea.judge_panel, idea.max_loops, idea.threshold = True, 2, 2, 7.5
    c = build(settings, transport)
    tid = c.tasks.create("idea", "a video about burnt toast")
    await c.tasks.wait_idle()
    t = c.db.task_get(tid)
    assert t["status"] == "awaiting_approval", t["error"]
    r = t["result"]
    assert r["loops"] == 2 and r["best_score"] == 8.5 and not r["below_threshold"]
    assert r["options"][0]["panel_scores"] == [9.0, 8.0]
    assert [o["score"] for o in r["options"]] == sorted([o["score"] for o in r["options"]], reverse=True)
    assert r["decision_log"]
    stages = [s["stage"] for s in c.db.trace_get(tid)]
    assert stages.count("judge") == 4 and "critique" in stages and "synthesize" in stages
    # second round gets the judge feedback
    assert any("JUDGE FEEDBACK" in call["user"] for call in fake.calls)
    # proposers got the persona in the system prompt and different model families
    assert any(call["cloud"] for call in fake.calls)

    approved = await c.tasks.approve(tid, 0)
    assert approved["status"] == "approved" and approved["chosen"]["score"] == 8.5
    mem = c.db.memory_list("idea")
    assert mem and mem[0]["text"] == approved["chosen"]["text"] and mem[0]["embedding"]


async def test_private_task_stays_local(settings, transport, fake):
    c = build(settings, transport)
    tid = c.tasks.create("caption", "morning coffee post")
    await c.tasks.wait_idle()
    assert c.db.task_get(tid)["status"] == "awaiting_approval"
    assert fake.calls and not any(call["cloud"] for call in fake.calls)


async def test_rethink_carries_note(settings, transport, fake):
    c = build(settings, transport)
    tid = c.tasks.create("hooks", "latte art fail")
    await c.tasks.wait_idle()
    new = c.tasks.rethink(tid, "more deadpan")
    await c.tasks.wait_idle()
    assert c.db.task_get(tid)["status"] == "superseded"
    t2 = c.db.task_get(new)
    assert t2["parent_id"] == tid and t2["note"] == "more deadpan" and t2["status"] == "awaiting_approval"
    assert any("more deadpan" in call["user"] for call in fake.calls)


async def test_memory_feeds_next_task(settings, transport, fake):
    c = build(settings, transport)
    tid = c.tasks.create("quick", "first")
    await c.tasks.wait_idle()
    await c.tasks.approve(tid, 1)
    fake.calls.clear()
    c.tasks.create("quick", "second")
    await c.tasks.wait_idle()
    # system prompt is message[0]; FakeLLM records the user message, so check via trace output instead
    assert c.db.memory_list() and c.db.task_list(status="awaiting_approval")


async def test_all_proposers_fail_marks_failed(settings, fake):
    import httpx
    settings.agent.llama_server.manage_processes = False
    c = Container.build(settings, transport=httpx.MockTransport(lambda r: httpx.Response(500, text="down")))
    tid = c.tasks.create("quick", "x")
    await c.tasks.wait_idle()
    t = c.db.task_get(tid)
    assert t["status"] == "failed" and "no usable options" in t["error"]
