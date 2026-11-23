from agent.bots.telegram_bot import format_task, format_trace


def test_format_pending_task_has_buttons_and_escapes():
    task = {"id": "abc123", "type": "hooks", "prompt": "<b>toast</b>", "note": None, "status": "awaiting_approval",
            "result": {"options": [{"text": "a & b", "score": 8.5, "rationale": "r"}, {"text": "c"}],
                       "best_score": 8.5, "threshold": 7.0, "loops": 1}}
    text, kb = format_task(task)
    assert "&lt;b&gt;toast" in text and "a &amp; b" in text and "[8.5]" in text
    datas = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert datas[:2] == ["a:abc123:0", "a:abc123:1"] and "t:abc123" in datas and all(len(d) <= 64 for d in datas)


def test_format_failed_and_trace():
    text, kb = format_task({"id": "x", "type": "quick", "prompt": "p", "status": "failed", "error": "boom"})
    assert "boom" in text and kb is None
    out = format_trace([{"stage": "propose", "loop": 0, "slot": "main", "role": "hook", "latency_ms": 1200,
                         "cost_usd": 0.01, "note": None}])
    assert "propose" in out and "$0.0100" in out
