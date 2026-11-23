"""Single-model variety (slot_profile: local) and the slot profile switch."""
import pytest
import yaml

from agent.config import apply_slot_profile, load_settings
from agent.container import Container
from agent.evals.harness import variant_cfg


def build(s, transport):
    return Container.build(s, transport=transport)


def test_shipped_profile_is_free_and_local(local_settings):
    a = local_settings.agent
    assert a.slot_profile == "local"
    for slot in ("main", "fast", "alt", "judge", "embed"):
        assert a.endpoints[a.slots[slot].endpoint].kind == "local", slot
    apply_slot_profile(a, "cloud")
    assert a.endpoints[a.slots["alt"].endpoint].kind == "cloud"
    assert "dolphin" in a.slots["alt"].model and "hermes" in a.slots["judge"].model
    with pytest.raises(ValueError):
        apply_slot_profile(a, "nope")


def test_missing_profile_fails_fast(root):
    cfg = yaml.safe_load((root / "config/agent.yaml").read_text())
    cfg["slot_profile"] = None
    (root / "config/agent.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="unknown slot"):
        load_settings(root)


async def test_local_idea_never_touches_cloud_and_varies(local_settings, transport, fake):
    c = build(local_settings, transport)
    tid = c.tasks.create("idea", "a weekly series")
    await c.tasks.wait_idle()
    t = c.db.task_get(tid)
    assert t["status"] == "awaiting_approval", t["error"]
    assert fake.calls and not any(call["cloud"] for call in fake.calls)

    props = [call for call in fake.calls if "Produce 5 distinct options" in call["user"]]
    assert len(props) == 4
    # role-specific sampling + a distinct seed per proposer
    seeds = [p["body"]["seed"] for p in props]
    assert len(set(seeds)) == 4
    hook = next(p for p in props if "first 1.5 seconds" in p["user"])
    assert hook["body"]["min_p"] == 0.05 and hook["body"]["presence_penalty"] == 0.4
    wild = next(p for p in props if "chaotic brainstormer" in p["user"])
    assert wild["model"] == "fast" and wild["body"]["temperature"] == 1.2
    # every proposer got a different creative constraint
    cons = [p["user"].split("CREATIVE CONSTRAINT FOR YOU (follow it): ")[1].split("\n")[0] for p in props]
    assert len(set(cons)) == 4
    # sequential anti-repetition: the first GPU proposer sees nothing taken, later ones do
    gpu = [p for p in props if p["model"] == "main"]
    assert "ANGLES ALREADY TAKEN" not in gpu[0]["user"]
    assert all("ANGLES ALREADY TAKEN" in p["user"] for p in gpu[1:])
    assert "main idea 0" in gpu[-1]["user"]

    stages = c.db.trace_get(tid)
    # identical texts from the same model got de-duplicated
    dd = [s for s in stages if s["stage"] == "dedupe"]
    assert dd and "kept" in dd[0]["note"]
    # audience panel: 2 judge calls with different personas
    judges = [call for call in fake.calls if "impartial judge" in call["user"]]
    assert len(judges) >= 2
    assert judges[0]["user"].split("Score as this viewer would react: ")[1].split(".")[0] != \
        judges[1]["user"].split("Score as this viewer would react: ")[1].split(".")[0]
    assert any((s["note"] or "").startswith("constraint:") for s in stages)


async def test_parallel_mode_has_no_taken_list(local_settings, transport, fake):
    local_settings.tasks.tasks["hooks"].diversify = "parallel"
    c = build(local_settings, transport)
    c.tasks.create("hooks", "toast")
    await c.tasks.wait_idle()
    props = [call for call in fake.calls if "distinct options" in call["user"]]
    assert props and not any("ANGLES ALREADY TAKEN" in p["user"] for p in props)


def test_no_variety_ablation(local_settings):
    base = local_settings.tasks.tasks["idea"]
    v = variant_cfg("pipeline:no-variety", base, set(local_settings.agent.slots))
    assert v.diversify == "parallel" and not v.constraints and v.dedupe is None and v.judge_panel == 1
    assert not any(p.startswith("fast@") for p in v.proposers)
    assert base.constraints and base.judge_panel == 2   # original untouched
