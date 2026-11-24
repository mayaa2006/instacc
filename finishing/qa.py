"""Quality checks: catch broken takes before anyone sees them.

Deterministic ffmpeg checks run on the RAW takes (before noise is added, which would hide frozen frames):
black stretches, frozen picture, wrong length, landscape output. An optional vision-model check looks at
sampled frames for a cinematic look, deformed hands, extra people and garbled text.
"""
from __future__ import annotations

import base64
from typing import Callable

from finishing import ff
from finishing.config import FinishingCfg

VLM_PROMPT = """These are frames from one take of a short vertical video that must look like a REAL phone video
filmed by one person. The take should show: {action}. Phone setup: {setup}.
Answer with ONLY JSON:
{{"looks_like_phone_video": true|false, "cinematic_signs": ["..."], "deformed_hands_or_face": true|false,
  "unexpected_people": true|false, "garbled_text": true|false, "matches_description": 1-5, "notes": "..."}}"""


def check_take(cfg: FinishingCfg, path: str, planned_s: float) -> list[str]:
    q, ffm = cfg.qa, cfg.ffmpeg_bin()
    issues: list[str] = []
    info = ff.probe(cfg.ffprobe_bin(), path)
    if info["height"] <= info["width"]:
        issues.append(f"not vertical ({info['width']}x{info['height']})")
    if planned_s and abs(info["duration"] - planned_s) > q.duration_tolerance_s:
        issues.append(f"length {info['duration']:.1f}s, planned {planned_s:g}s")
    log = ff.detect(ffm, path, vf=f"blackdetect=d={q.black_max_s}:pix_th=0.10,"
                                   f"freezedetect=n=-55dB:d={q.frozen_max_s}")
    for s, e in ff.spans(log, "black_start", "black_end"):
        issues.append(f"black picture at {s:.1f}s")
    for s, _ in ff.spans(log, "freeze_start", "freeze_end"):
        issues.append(f"frozen picture from {s:.1f}s")
    return issues


def sample_frames(cfg: FinishingCfg, path: str, n: int, duration: float) -> list[bytes]:
    import subprocess
    frames = []
    for i in range(n):
        t = duration * (i + 0.5) / n
        p = subprocess.run([cfg.ffmpeg_bin(), "-v", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
                            "-vf", "scale=-2:768", "-f", "image2", "-c:v", "mjpeg", "-"], capture_output=True, timeout=60)
        if p.returncode == 0 and p.stdout:
            frames.append(p.stdout)
    return frames


def vlm_messages(frames: list[bytes], shot: dict) -> list[dict]:
    content = [{"type": "text", "text": VLM_PROMPT.format(action=shot.get("action", ""), setup=shot.get("setup", ""))}]
    content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(f).decode()}}
                for f in frames]
    return [{"role": "user", "content": content}]


def vlm_issues(verdict: dict) -> list[str]:
    out = []
    if verdict.get("looks_like_phone_video") is False:
        out.append("doesn't look like a phone video" + (f" ({', '.join(verdict.get('cinematic_signs') or [])})"
                                                         if verdict.get("cinematic_signs") else ""))
    if verdict.get("deformed_hands_or_face"):
        out.append("deformed hands or face")
    if verdict.get("unexpected_people"):
        out.append("unexpected people in frame")
    if verdict.get("garbled_text"):
        out.append("garbled text in the picture")
    try:
        if int(verdict.get("matches_description", 5)) <= 2:
            out.append("doesn't match the shot description")
    except (TypeError, ValueError):
        pass
    return out


VlmCheck = Callable[[list[dict]], dict]   # messages -> parsed JSON verdict
