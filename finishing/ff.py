"""Small ffmpeg/ffprobe helpers (blocking; the agent calls the pipeline in a worker thread)."""
from __future__ import annotations

import json
import re
import subprocess


class FFError(RuntimeError):
    pass


def run(cmd: list[str], timeout: float = 900) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise FFError(f"{cmd[0]} failed ({p.returncode}): {p.stderr[-1200:]}")
    return p.stderr


def probe(ffprobe: str, path: str) -> dict:
    out = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise FFError(f"ffprobe failed on {path}: {out.stderr[-400:]}")
    d = json.loads(out.stdout)
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), None)
    num, _, den = (v.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {"duration": float(d.get("format", {}).get("duration") or 0), "width": int(v.get("width") or 0),
            "height": int(v.get("height") or 0), "fps": round(fps, 2), "has_audio": a is not None}


def detect(ffmpeg: str, path: str, vf: str | None = None, af: str | None = None) -> str:
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-i", path]
    cmd += ["-vf", vf] if vf else ["-vn"]
    cmd += ["-af", af] if af else ["-an"]
    cmd += ["-f", "null", "-"]
    return run(cmd)


def spans(log: str, start_key: str, end_key: str) -> list[tuple[float, float]]:
    starts = [float(x) for x in re.findall(rf"{start_key}:\s*(-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(rf"{end_key}:\s*(-?[\d.]+)", log)]
    return list(zip(starts, ends + [None] * (len(starts) - len(ends))))
