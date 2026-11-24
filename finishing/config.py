from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent


class Output(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: int = 30
    interpolate: bool = False
    crf: int = 21
    audio_bitrate: str = "160k"


class Look(BaseModel):
    noise: float = 5
    exposure_drift: float = 0.012
    wobble_px: float = 9
    front_camera_softness: float = 0.45
    mirror_selfies: bool = False


class Captions(BaseModel):
    enabled: bool = True
    font: str = "auto"
    size: float = 0.042
    y: float = 0.27
    max_chars_per_line: int = 22
    max_chars: int = 90


class Voice(BaseModel):
    engine: str = "none"
    piper_binary: str = "tools/piper/piper.exe"
    piper_voice: str = ""
    command: str = ""
    voice_ref: str = ""
    max_speedup: float = 1.15


class Audio(BaseModel):
    phone_mic: bool = True
    room_tone: bool = True
    room_tone_level: float = 0.004
    room_tones: dict[str, str] | None = None
    dialogue_gain_db: float = 0


class LipSync(BaseModel):
    enabled: bool = False
    command: str = ""
    timeout_s: float = 900


class Vlm(BaseModel):
    enabled: bool = False
    frames_per_take: int = 3


class QA(BaseModel):
    black_max_s: float = 0.4
    frozen_max_s: float = 0.9
    duration_tolerance_s: float = 1.5
    vlm: Vlm = Field(default_factory=Vlm)
    auto_rerender_attempts: int = 2


class FinishingCfg(BaseModel):
    ffmpeg: str = "auto"
    output: Output = Field(default_factory=Output)
    look: Look = Field(default_factory=Look)
    captions: Captions = Field(default_factory=Captions)
    voice: Voice = Field(default_factory=Voice)
    audio: Audio = Field(default_factory=Audio)
    lipsync: LipSync = Field(default_factory=LipSync)
    qa: QA = Field(default_factory=QA)
    root: Path = ROOT

    def path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else self.root / q

    def ffmpeg_bin(self) -> str:
        if self.ffmpeg and self.ffmpeg != "auto":
            return str(self.path(self.ffmpeg))
        exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        found = sorted((self.root / "tools").glob(f"ffmpeg*/**/{exe}"))
        if found:
            return str(found[0])
        return shutil.which("ffmpeg") or "ffmpeg"

    def ffprobe_bin(self) -> str:
        ff = Path(self.ffmpeg_bin())
        probe = ff.with_name(ff.name.replace("ffmpeg", "ffprobe"))
        return str(probe) if probe.exists() else (shutil.which("ffprobe") or "ffprobe")

    def font_file(self) -> str | None:
        if self.captions.font and self.captions.font != "auto":
            return str(self.path(self.captions.font))
        for c in ("C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"):
            if Path(c).exists():
                return c
        return None


def load_finishing(root: Path | None = None) -> FinishingCfg:
    root = Path(root or ROOT)
    p = root / "config" / "finishing.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    cfg = FinishingCfg.model_validate(data or {})
    cfg.root = root
    return cfg
