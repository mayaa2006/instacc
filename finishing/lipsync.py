"""Lip-sync for takes where her face speaks to the camera, via a local model run as a command."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from finishing.config import FinishingCfg

FACE_SETUPS = {"selfie", "propped", "mirror"}


def needs_lipsync(setup: str, dialogue: str) -> bool:
    return setup in FACE_SETUPS and bool((dialogue or "").strip())


def lipsync(cfg: FinishingCfg, video: Path, audio: Path, out: Path) -> Path:
    ls = cfg.lipsync
    if not ls.enabled or not ls.command:
        raise RuntimeError("lip-sync not configured")
    cmd = ls.command.format(video=str(video), audio=str(audio), out=str(out))
    p = subprocess.run(shlex.split(cmd, posix=not cmd[:3].lower().endswith(".exe")), capture_output=True,
                       timeout=ls.timeout_s, cwd=str(cfg.root))
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(f"lip-sync failed: {p.stderr[-400:]!r}")
    return out
