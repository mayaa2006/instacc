"""Phone-camera look as an ffmpeg filter chain, per take and per phone setup."""
from __future__ import annotations

import random
from pathlib import Path

from finishing.config import FinishingCfg

HANDHELD = {"selfie", "pov", "mirror"}
FRONT_CAMERA = {"selfie", "mirror"}


def wrap_caption(text: str, per_line: int, max_chars: int) -> str:
    text = " ".join(text.split())[:max_chars]
    lines, cur = [], ""
    for w in text.split(" "):
        if cur and len(cur) + 1 + len(w) > per_line:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _esc(path: str) -> str:
    """Escape a path for use inside an ffmpeg filter argument (Windows drive colons included)."""
    return path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def video_filter(cfg: FinishingCfg, setup: str, seed: int, caption_file: Path | None) -> str:
    o, lk = cfg.output, cfg.look
    rnd = random.Random(seed)
    f: list[str] = []
    wob = lk.wobble_px if setup in HANDHELD else 0
    pad = int(wob * 2 + 2) if wob else 0
    W, H = o.width + 2 * pad, o.height + 2 * pad
    f.append(f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos")
    if wob:   # two slow sines per axis = natural hand drift, never perfectly still
        a1, a2 = rnd.uniform(0.6, 1.0) * wob, rnd.uniform(0.2, 0.45) * wob
        p1, p2, p3, p4 = (rnd.uniform(0, 6.28) for _ in range(4))
        f1, f2 = rnd.uniform(0.35, 0.6), rnd.uniform(1.1, 1.7)
        f.append(f"crop={o.width}:{o.height}:"
                 f"x='(iw-{o.width})/2+{a1:.2f}*sin(2*PI*t*{f1:.2f}+{p1:.2f})+{a2:.2f}*sin(2*PI*t*{f2:.2f}+{p2:.2f})':"
                 f"y='(ih-{o.height})/2+{a1 * 0.8:.2f}*sin(2*PI*t*{f1 * 0.9:.2f}+{p3:.2f})+{a2:.2f}*sin(2*PI*t*{f2 * 1.1:.2f}+{p4:.2f})'")
    else:
        f.append(f"crop={o.width}:{o.height}")
    if setup in FRONT_CAMERA and lk.mirror_selfies:
        f.append("hflip")
    if setup in FRONT_CAMERA and lk.front_camera_softness > 0:
        f.append(f"gblur=sigma={lk.front_camera_softness:.2f}")
    f.append(f"minterpolate=fps={o.fps}:mi_mode=mci" if o.interpolate else f"fps={o.fps}")
    if lk.exposure_drift > 0:
        per = rnd.uniform(2.2, 3.6)
        f.append(f"eq=brightness='{lk.exposure_drift:.4f}*sin(2*PI*t/{per:.2f})':eval=frame")
    if lk.noise > 0:
        f.append(f"noise=alls={lk.noise:g}:allf=t")
    font = cfg.font_file()
    if caption_file and font and cfg.captions.enabled:
        c = cfg.captions
        f.append(f"drawtext=fontfile='{_esc(font)}':textfile='{_esc(str(caption_file))}':"
                 f"fontsize={int(o.height * c.size)}:fontcolor=white:borderw={max(2, int(o.height * 0.004))}:"
                 f"bordercolor=black@0.85:line_spacing={int(o.height * 0.008)}:x=(w-text_w)/2:y=h*{c.y}")
    f += ["setsar=1", "format=yuv420p"]
    return ",".join(f)
