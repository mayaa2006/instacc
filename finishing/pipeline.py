"""rendered takes + VideoJob -> finished phone-realistic mp4 + QA report."""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from finishing import ff, qa
from finishing.audio import VoiceUnavailable, speak, take_audio
from finishing.config import FinishingCfg, load_finishing
from finishing.lipsync import lipsync, needs_lipsync
from finishing.look import video_filter, wrap_caption

log = logging.getLogger(__name__)


@dataclass
class FinishResult:
    path: str | None
    passed: bool
    rerender: list[int] = field(default_factory=list)      # 0-based shot indices to render again
    take_issues: dict[int, list[str]] = field(default_factory=dict)
    final_issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)          # things that worked around a missing tool etc.

    def as_dict(self) -> dict:
        return {"path": self.path, "passed": self.passed, "rerender": self.rerender,
                "take_issues": {str(k): v for k, v in self.take_issues.items()},
                "final_issues": self.final_issues, "notes": self.notes}


def finish(job: dict, takes: list[str], out: str, cfg: FinishingCfg | None = None, *,
           stop_on_bad_takes: bool = False, vlm_check: qa.VlmCheck | None = None,
           timing: dict | None = None) -> FinishResult:
    cfg = cfg or load_finishing()
    shots = job.get("shots") or []
    if len(takes) != len(shots):
        return FinishResult(None, False, final_issues=[f"{len(takes)} takes for {len(shots)} shots"])
    res = FinishResult(None, True)

    # 1. QA on raw takes
    for i, (t, sh) in enumerate(zip(takes, shots)):
        if not Path(t).exists():
            res.take_issues[i] = ["file missing"]
            continue
        issues = qa.check_take(cfg, t, float(sh.get("duration_s") or 0))
        if vlm_check and cfg.qa.vlm.enabled:
            try:
                info = ff.probe(cfg.ffprobe_bin(), t)
                frames = qa.sample_frames(cfg, t, cfg.qa.vlm.frames_per_take, info["duration"])
                issues += qa.vlm_issues(vlm_check(qa.vlm_messages(frames, sh)))
            except Exception as e:
                res.notes.append(f"take {i + 1}: vision check skipped ({e})")
        if issues:
            res.take_issues[i] = issues
    res.rerender = sorted(res.take_issues)
    if res.rerender:
        res.passed = False
        if stop_on_bad_takes:
            return res

    # 2. per-take voice, lip-sync, look, captions
    work = Path(tempfile.mkdtemp(prefix="finish_"))
    try:
        ffm = cfg.ffmpeg_bin()
        location = (job.get("continuity") or {}).get("location", "")
        parts: list[Path] = []
        for i, (t, sh) in enumerate(zip(takes, shots)):
            if not Path(t).exists():
                continue
            setup = (sh.get("setup") or "").lower()
            dur = ff.probe(cfg.ffprobe_bin(), t)["duration"] or float(sh.get("duration_s") or 4)
            video = Path(t)
            voice = None
            line = (sh.get("dialogue") or "").strip()
            if line and cfg.voice.engine != "none":
                try:
                    voice = speak(cfg, line, work / f"voice{i}.wav", work)
                except VoiceUnavailable as e:
                    res.notes.append(f"take {i + 1}: no voice ({e})")
            if voice is not None and cfg.lipsync.enabled and needs_lipsync(setup, line):
                try:
                    video = lipsync(cfg, video, voice, work / f"lipsync{i}.mp4")
                except Exception as e:
                    res.notes.append(f"take {i + 1}: lip-sync skipped ({e})")
            wav, anotes = take_audio(cfg, dur, voice, location, work / f"audio{i}.wav")
            res.notes += [f"take {i + 1}: {n}" for n in anotes]
            cap = None
            text = (sh.get("on_screen_text") or "").strip()
            if text and cfg.captions.enabled:
                cap = work / f"cap{i}.txt"
                cap.write_text(wrap_caption(text, cfg.captions.max_chars_per_line, cfg.captions.max_chars),
                               encoding="utf-8")
                if not cfg.font_file():
                    res.notes.append("captions skipped: no font found (set captions.font)")
            vf = video_filter(cfg, setup, seed=hash((job.get("post_id"), i)) & 0xFFFF, caption_file=cap)
            part = work / f"part{i:02d}.mp4"
            ff.run([ffm, "-y", "-hide_banner", "-i", str(video), "-i", str(wav), "-filter:v", vf,
                    "-map", "0:v:0", "-map", "1:a:0", "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "16", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", str(part)])
            parts.append(part)
        if not parts:
            res.passed = False
            res.final_issues.append("no usable takes")
            return res

        # 3. hard-cut concat + phone-like export
        listf = work / "list.txt"
        listf.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        o = cfg.output
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        ff.run([ffm, "-y", "-hide_banner", "-f", "concat", "-safe", "0", "-i", str(listf),
                "-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-crf", str(o.crf), "-r", str(o.fps),
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", o.audio_bitrate, "-ar", "48000", "-ac", "2",
                "-movflags", "+faststart", out])
        res.path = out

        # 4. QA on the result
        info = ff.probe(cfg.ffprobe_bin(), out)
        if (info["width"], info["height"]) != (o.width, o.height):
            res.final_issues.append(f"size {info['width']}x{info['height']}")
        if abs(info["fps"] - o.fps) > 0.5:
            res.final_issues.append(f"{info['fps']} fps")
        if not info["has_audio"]:
            res.final_issues.append("no audio track")
        t = timing or {}
        if t and not t.get("min_total_s", 0) - 0.5 <= info["duration"] <= t.get("max_total_s", 1e9) + 0.5:
            res.final_issues.append(f"video is {info['duration']:.1f}s")
        if res.final_issues:
            res.passed = False
        return res
    finally:
        shutil.rmtree(work, ignore_errors=True)
