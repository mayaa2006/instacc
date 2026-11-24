"""Phone-realism: shot lists must be filmable by one person with one phone."""
import pytest

from agent.container import Container
from agent.content.realism import check_shots, render_prompt


def test_style_loaded(settings):
    st = settings.style
    assert settings.style_file.active == "phone_ugc"
    assert {"selfie", "propped", "pov"} <= set(st.setups)
    assert "drone" in st.banned_terms and "cinematic" in st.negative_prompt


@pytest.mark.parametrize("shot,expect", [
    ({"setup": "selfie", "action": "she pours milk with both hands"}, "both hands"),
    ({"setup": "pov", "action": "she smiles at the camera while drawing"}, "face isn't visible"),
    ({"setup": "propped", "camera": "drone shot over the city", "action": "she waves"}, "'drone'"),
    ({"setup": "propped", "action": "a friend films her from across the street"}, "nobody else"),
    ({"setup": "", "action": "x"}, "must be one of"),
    ({"setup": "propped", "action": "she pours milk with both hands", "camera_motion": "static"}, None),
    ({"setup": "pov", "action": "her hands tilt the jug, the heart collapses into a blob"}, None),
    ({"setup": "selfie", "action": "she rolls her eyes and says attempt ten into the lens"}, None),
])
def test_check_shots(settings, shot, expect):
    shot = {"duration_s": 5, **shot}
    filler = {"setup": "propped", "duration_s": 5, "action": "she sips the coffee"}
    issues = check_shots([shot, filler], settings.style)
    if expect is None:
        assert issues == []
    else:
        assert any(expect in i for i in issues), issues


def test_render_prompt_has_phone_look(settings):
    p = render_prompt({"setup": "pov", "action": "her hands pour milk into a cup", "setting": "kitchen counter",
                       "camera_motion": "slight handheld shake"},
                      {"outfit": "cream knit sweater", "lighting": "morning window light"}, settings.style, "auburn hair")
    assert "first-person POV smartphone video" in p and "shot on a smartphone" in p and "cream knit sweater" in p
    assert "auburn hair" not in p   # face isn't visible in POV, so no face description


async def test_producer_rewrites_unrealistic_shots(settings, transport, fake):
    settings.agent.llama_server.manage_processes = False
    fake.produce_unrealistic = 1   # first answer has a drone shot, a POV smile and a missing setup
    c = Container.build(settings, transport=transport)
    p = c.posts.create("burnt toast", auto_script=False)
    p = await c.posts.generate_script(p["id"])
    assert p["status"] == "script_review", p["error"]
    job = p["video_job"]
    assert job["schema_version"] == "0.2" and job["style"] == "phone_ugc"
    assert [s["setup"] for s in job["shots"]] == ["selfie", "propped"]
    assert all("smartphone" in s["render_prompt"] for s in job["shots"])
    assert "drone" in job["negative_prompt"] and job["realism_warnings"] == []
    assert "smartphone" in job["shots"][0]["keyframe_prompt"]
    trace = c.db.trace_get(f"post:{p['id']}")
    stages = [(t["stage"], t["note"]) for t in trace]
    fixes = [t["output"] for t in trace if (t["note"] or "").startswith("auto-fixed")]
    assert fixes and "pov → selfie" in fixes[0] and "missing → " in fixes[0]   # POV smile -> selfie, setup chosen
    # the drone can't be auto-fixed, so the model was asked to rewrite, with the reason
    retry = [call for call in fake.calls if "can't be filmed by one person" in call["user"]]
    assert retry and "'drone'" in retry[0]["user"] and "face isn't visible" not in retry[0]["user"]


async def test_unfixable_shots_are_flagged_not_hidden(settings, transport, fake):
    settings.agent.llama_server.manage_processes = False
    fake.produce_unrealistic = 99   # never fixes it
    c = Container.build(settings, transport=transport)
    p = c.posts.create("burnt toast", auto_script=False)
    p = await c.posts.generate_script(p["id"])
    assert p["status"] == "script_review" and p["video_job"]["realism_warnings"]
    from agent.content.producer import summarize_job
    assert "⚠️" in summarize_job(p["video_job"])


async def test_ideas_know_the_filming_style(settings, transport, fake):
    settings.agent.llama_server.manage_processes = False
    c = Container.build(settings, transport=transport)
    c.tasks.create("hooks", "toast")
    await c.tasks.wait_idle()
    sys0 = fake.calls[0]["system"]
    assert "HOW VIDEOS ARE MADE" in sys0 and "one smartphone" in sys0 and "POV hands" in sys0
    judge = [call for call in fake.calls if "impartial judge" in call["user"]]
    assert judge and "Filmable and real" in judge[0]["user"]


def test_timing_rules(settings):
    one = [{"setup": "selfie", "duration_s": 4, "action": "she sighs"}]
    issues = check_shots(one, settings.style)
    assert any("2-6 shots" in i for i in issues) and any("10-30 seconds" in i for i in issues)
    long_take = [{"setup": "selfie", "duration_s": 12, "action": "she talks"}, {"setup": "pov", "duration_s": 4, "action": "her hands"}]
    assert any("shot 1: each take" in i for i in check_shots(long_take, settings.style))


def test_auto_repair(settings):
    from agent.content.realism import auto_repair
    shots = [{"setup": "pov", "duration_s": 5, "action": "she smiles at the camera"},
             {"setup": "selfie", "duration_s": 5, "action": "she pours milk with both hands"},
             {"setup": "", "duration_s": 5, "action": "her hands stir the cup"}]
    fixed, notes = auto_repair(shots, settings.style)
    assert [s["setup"] for s in fixed] == ["selfie", "propped", "pov"] and len(notes) == 3
    assert check_shots(fixed, settings.style) == []


def test_auto_repair_timing(settings):
    from agent.content.realism import auto_repair
    short = [{"setup": "selfie", "duration_s": 4, "action": "she sighs"}, {"setup": "pov", "duration_s": 4, "action": "her hands"}]
    fixed, notes = auto_repair(short, settings.style)
    assert sum(s["duration_s"] for s in fixed) >= 10 and check_shots(fixed, settings.style) == []
    too_long = [{"setup": "selfie", "duration_s": 12, "action": "she talks"}] * 3
    fixed, _ = auto_repair(too_long, settings.style)
    assert all(s["duration_s"] <= 5 for s in fixed) and check_shots(fixed, settings.style) == []
    one = [{"setup": "selfie", "duration_s": 4, "action": "she sighs"}]
    assert check_shots(auto_repair(one, settings.style)[0], settings.style)   # too few shots: model must rewrite
