"""Finishing toolkit with real ffmpeg (skipped if ffmpeg isn't installed)."""
import json
import shutil
import subprocess

import pytest

from finishing import ff
from finishing.config import load_finishing
from finishing.look import wrap_caption
from finishing.pipeline import finish

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def make_take(path, src="testsrc2", dur=5, extra_vf=None):
    vf = ["-vf", extra_vf] if extra_vf else []
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"{src}=s=720x1280:r=16:d={dur}", *vf,
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


@pytest.fixture
def cfg(tmp_path):
    c = load_finishing()
    c.voice.engine = "command"
    c.voice.command = f"ffmpeg -v error -y -f lavfi -i sine=frequency=220:duration=1.6 {{out}}"
    return c


JOB = {"post_id": "t1", "continuity": {"location": "kitchen"},
       "shots": [{"setup": "selfie", "duration_s": 5, "dialogue": "okay so. attempt ten.",
                  "on_screen_text": "latte art attempt #10"},
                 {"setup": "pov", "duration_s": 5, "dialogue": "", "on_screen_text": ""}]}


def test_wrap_caption():
    assert wrap_caption("latte art attempt #10 (it did not go well)", 22, 90) == \
        "latte art attempt #10\n(it did not go well)"
    assert len(wrap_caption("x " * 100, 22, 90).replace("\n", " ")) <= 90


def test_finish_makes_phone_video(tmp_path, cfg):
    takes = [make_take(tmp_path / "a.mp4"), make_take(tmp_path / "b.mp4", "smptehdbars", extra_vf="noise=alls=20:allf=t")]
    out = tmp_path / "final.mp4"
    r = finish(JOB, takes, str(out), cfg, timing={"min_total_s": 10, "max_total_s": 30})
    assert r.passed and r.path == str(out), r.as_dict()
    info = ff.probe(cfg.ffprobe_bin(), str(out))
    assert (info["width"], info["height"], info["fps"]) == (1080, 1920, 30.0) and info["has_audio"]
    assert 9.5 <= info["duration"] <= 10.6
    vol = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(out), "-af", "volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    peak = float(vol.split("max_volume:")[1].split("dB")[0])
    assert peak > -18, f"voice too quiet: {peak} dB"   # a pure test tone at -16 LUFS peaks around -14 dB


def test_qa_catches_black_and_frozen(tmp_path, cfg):
    black = make_take(tmp_path / "black.mp4", "testsrc2", extra_vf="drawbox=enable='between(t,1,2.5)':c=black:t=fill")
    frozen = make_take(tmp_path / "frozen.mp4", "testsrc2", extra_vf="trim=0:0.5,tpad=stop_mode=clone:stop_duration=4.5")
    short = make_take(tmp_path / "short.mp4", dur=2)
    r = finish({"shots": [{"setup": "pov", "duration_s": 5}] * 3}, [black, frozen, short], str(tmp_path / "x.mp4"),
               cfg, stop_on_bad_takes=True)
    assert not r.passed and r.rerender == [0, 1, 2] and r.path is None
    assert any("black" in i for i in r.take_issues[0])
    assert any("frozen" in i for i in r.take_issues[1])
    assert any("length" in i for i in r.take_issues[2])


def test_wrong_take_count(tmp_path, cfg):
    r = finish(JOB, [make_take(tmp_path / "a.mp4")], str(tmp_path / "x.mp4"), cfg)
    assert not r.passed and "1 takes for 2 shots" in r.final_issues[0]


def test_missing_voice_engine_is_a_note_not_a_failure(tmp_path, cfg):
    cfg.voice.engine = "piper"   # binary not installed here
    takes = [make_take(tmp_path / "a.mp4"), make_take(tmp_path / "b.mp4")]
    r = finish(JOB, takes, str(tmp_path / "f.mp4"), cfg)
    assert r.passed and any("no voice" in n for n in r.notes)


def test_vlm_check_flags_issues(tmp_path, cfg):
    cfg.qa.vlm.enabled = True
    seen = []

    def vlm(messages):
        seen.append(messages)
        return {"looks_like_phone_video": False, "cinematic_signs": ["shallow depth of field"],
                "deformed_hands_or_face": True, "matches_description": 4}
    takes = [make_take(tmp_path / "a.mp4"), make_take(tmp_path / "b.mp4")]
    r = finish(JOB, takes, str(tmp_path / "f.mp4"), cfg, stop_on_bad_takes=True, vlm_check=vlm)
    assert r.rerender == [0, 1] and "deformed hands or face" in r.take_issues[0]
    imgs = [c for c in seen[0][0]["content"] if c["type"] == "image_url"]
    assert len(imgs) == 3 and imgs[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------- agent integration
async def test_agent_finishes_takes_and_rerenders_bad_ones(settings, transport, fake, tmp_path):
    from agent.container import Container
    settings.agent.llama_server.manage_processes = False
    c = Container.build(settings, transport=transport)
    c.posts.finishing_cfg.voice.engine = "none"
    p = c.posts.create("burnt toast", auto_script=False)
    p = await c.posts.generate_script(p["id"])
    await c.posts.approve_script(p["id"])
    c.posts.claim(p["id"])
    good = [make_take(tmp_path / f"g{i}.mp4", extra_vf="noise=alls=20:allf=t") for i in range(2)]
    frozen = make_take(tmp_path / "f.mp4", extra_vf="trim=0:0.5,tpad=stop_mode=clone:stop_duration=4.5")

    # a frozen take goes back to the render queue, with the reason and which take to redo
    p = await c.posts.rendered(p["id"], takes=[good[0], frozen])
    assert p["status"] == "render_queued" and p["qa"]["rerender"] == [1] and "frozen" in p["error"]
    assert [q["id"] for q in c.posts.render_queue()] == [p["id"]]

    c.posts.claim(p["id"])
    p = await c.posts.rendered(p["id"], takes=good)
    assert p["status"] == "captioning", p.get("error")
    info = __import__("finishing.ff", fromlist=["probe"]).probe(c.posts.finishing_cfg.ffprobe_bin(), p["media_path"])
    assert p["media_path"].endswith(f"media/{p['id']}.mp4") and (info["width"], info["height"]) == (1080, 1920)
    assert p["qa"]["passed"] and p["qa"]["attempts"] == 1


async def test_rerender_gives_up_after_max_attempts(settings, transport, fake, tmp_path):
    from agent.container import Container
    settings.agent.llama_server.manage_processes = False
    c = Container.build(settings, transport=transport)
    c.posts.finishing_cfg.voice.engine = "none"
    c.posts.finishing_cfg.qa.auto_rerender_attempts = 1
    p = c.posts.create("x", auto_script=False)
    p = await c.posts.generate_script(p["id"])
    await c.posts.approve_script(p["id"])
    frozen = make_take(tmp_path / "f.mp4", extra_vf="trim=0:0.5,tpad=stop_mode=clone:stop_duration=4.5")
    ok = make_take(tmp_path / "ok.mp4", extra_vf="noise=alls=20:allf=t")
    p = await c.posts.rendered(p["id"], takes=[ok, frozen])
    assert p["status"] == "render_queued"
    p = await c.posts.rendered(p["id"], takes=[ok, frozen])   # still bad: finish anyway and show the user
    assert p["status"] == "captioning" and p["qa"]["take_issues"]["1"]
