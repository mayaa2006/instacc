import pytest

from agent.container import Container
from agent.evals.harness import EvalError, load_cases, variant_cfg

CASES = [
    {"id": "a", "type": "hooks", "prompt": "toast"},
    {"id": "b", "type": "idea", "prompt": "series"},
]


def build(settings, transport):
    settings.agent.llama_server.manage_processes = False
    return Container.build(settings, transport=transport)


def test_golden_set_valid(settings):
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    cases = load_cases(root)
    assert len(cases) >= 30 and len({c["id"] for c in cases}) == len(cases)
    assert {c["type"] for c in cases} <= set(settings.tasks.tasks)


def test_variant_cfgs(settings):
    base = settings.tasks.tasks["idea"]
    base.critique = True
    slots = set(settings.agent.slots)
    single = variant_cfg("single:judge", base, slots)
    assert single.proposers == ["judge@0.7:writer"] and single.judge is None and single.n_options == 1
    abl = variant_cfg("pipeline:no-critique,no-loops", base, slots)
    assert not abl.critique and abl.max_loops == 0 and abl.judge == base.judge
    assert base.critique  # base config untouched
    for bad in ("single:nope", "pipeline:no-magic", "random"):
        with pytest.raises(EvalError):
            variant_cfg(bad, base, slots)


async def test_eval_run_auto_judge_and_report(settings, transport, fake):
    c = build(settings, transport)
    rid = c.evals.start(["single:main", "pipeline"], cases=CASES, auto_judge="judge")
    await c.evals.wait_idle()
    run = c.evals.get_run(rid)
    assert run["status"] == "done", run["error"]
    rep = c.evals.report(rid)
    rows = {r["variant"]: r for r in rep["variants"]}
    # fake judge prefers synthesized output in both orders -> pipeline wins every pair
    assert rows["pipeline"]["wins"] == 2 and rows["single:main"]["losses"] == 2
    assert rows["pipeline"]["win_rate"] == 1.0 and rep["variants"][0]["variant"] == "pipeline"
    assert rep["head_to_head"]["pipeline"]["single:main"]["win_rate"] == 1.0
    assert rows["single:main"]["avg_wall_s"] is not None
    # eval tasks don't pollute the approval queue
    assert c.db.task_list() == []


async def test_position_inconsistent_judge_is_tie(settings, transport, fake):
    c = build(settings, transport)
    # both variants single -> no "final" -> fake judge says tie; identical outputs short-circuit to tie too
    rid = c.evals.start(["single:main", "single:fast"], cases=CASES[:1], auto_judge="judge")
    await c.evals.wait_idle()
    rep = c.evals.report(rid)
    assert all(r["ties"] == 1 for r in rep["variants"])


async def test_blind_human_voting(settings, transport, fake):
    c = build(settings, transport)
    rid = c.evals.start(["single:main", "pipeline"], cases=CASES, auto_judge=None)
    await c.evals.wait_idle()
    seen = 0
    while (pair := c.evals.next_pair(rid)) is not None:
        assert "variant" not in str(pair.keys()) and set(pair) >= {"token", "left", "right", "prompt"}
        choice = "left" if "final" in pair["left"] else "right"
        c.evals.vote(pair["token"], choice)
        seen += 1
    assert seen == 2
    with pytest.raises(EvalError):
        c.evals.vote("bogus", "left")
    rep = c.evals.report(rid, voter="human")
    assert {r["variant"]: r["win_rate"] for r in rep["variants"]} == {"pipeline": 1.0, "single:main": 0.0}
    assert c.evals.report(rid, voter="llm")["votes"] == 0


def test_start_validation(settings, transport):
    c = build(settings, transport)
    with pytest.raises(EvalError):
        c.evals.start(["pipeline"], cases=CASES)
    with pytest.raises(EvalError):
        c.evals.start(["pipeline", "single:main"], cases=CASES, auto_judge="nope")
