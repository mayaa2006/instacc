"""Voice (TTS), phone-mic sound and room tone for each take."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from finishing import ff
from finishing.config import FinishingCfg


class VoiceUnavailable(RuntimeError):
    pass


def speak(cfg: FinishingCfg, text: str, out: Path, work: Path) -> Path:
    """Render `text` to a wav with the configured TTS engine."""
    v = cfg.voice
    tf = work / f"{out.stem}.txt"
    tf.write_text(text, encoding="utf-8")
    if v.engine == "piper":
        voice = cfg.path(v.piper_voice)
        if v.piper_binary == "module":      # pip install piper-tts into the portable Python (get-voice.bat)
            import importlib.util
            import sys
            if importlib.util.find_spec("piper") is None:
                raise VoiceUnavailable("piper-tts isn't installed (run get-voice.bat)")
            exe = [sys.executable, "-m", "piper"]
        else:
            binary = cfg.path(v.piper_binary)
            if not binary.exists():
                raise VoiceUnavailable(f"piper not found at {binary}")
            exe = [str(binary)]
        if not voice.exists():
            raise VoiceUnavailable(f"voice model missing ({voice}); run get-voice.bat")
        with open(tf, "rb") as fin:
            p = subprocess.run([*exe, "--model", str(voice), "--output_file", str(out)], stdin=fin,
                               capture_output=True, timeout=300)
        if p.returncode != 0:
            raise VoiceUnavailable(f"piper failed: {p.stderr[-300:]!r}")
    elif v.engine == "command":
        if not v.command:
            raise VoiceUnavailable("voice.command is empty")
        cmd = v.command.format(text_file=str(tf), out=str(out), voice_ref=str(cfg.path(v.voice_ref)) if v.voice_ref else "")
        p = subprocess.run(shlex.split(cmd, posix=not cmd[:3].lower().endswith(".exe")), capture_output=True,
                           timeout=600, cwd=str(cfg.root))
        if p.returncode != 0:
            raise VoiceUnavailable(f"voice command failed: {p.stderr[-300:]!r}")
    else:
        raise VoiceUnavailable("no voice engine configured")
    if not out.exists() or out.stat().st_size < 1000:
        raise VoiceUnavailable("voice engine produced no audio")
    return out


def room_tone_file(cfg: FinishingCfg, location: str) -> Path | None:
    for key, rel in (cfg.audio.room_tones or {}).items():
        if key.lower() in (location or "").lower():
            p = cfg.path(rel)
            if p.exists():
                return p
    return None


def take_audio(cfg: FinishingCfg, duration: float, voice: Path | None, location: str, out: Path) -> tuple[Path, list[str]]:
    """Mix voice (phone-mic coloured, fitted to the take) over room tone. Returns (wav, notes)."""
    a = cfg.audio
    ffm = cfg.ffmpeg_bin()
    notes: list[str] = []
    inputs: list[str] = []
    chains: list[str] = []
    labels: list[str] = []
    n = 0
    if voice is not None:
        vlen = ff.probe(cfg.ffprobe_bin(), str(voice))["duration"]
        tempo = 1.0
        if vlen > duration - 0.3:
            tempo = min(cfg.voice.max_speedup, max(1.0, vlen / max(0.5, duration - 0.3)))
            if vlen / tempo > duration:
                notes.append(f"line is {vlen:.1f}s for a {duration:.1f}s take; trimmed")
        chain = [f"atempo={tempo:.3f}"] if tempo > 1.0 else []
        if a.phone_mic:   # small phone mic: no deep lows, rolled-off highs, compressed, a touch of room
            chain += ["highpass=f=140", "lowpass=f=7800", "acompressor=threshold=-20dB:ratio=3:attack=8:release=120",
                      "aecho=0.8:0.6:18:0.12"]
        chain.append("loudnorm=I=-16:TP=-2:LRA=9")   # social-video loudness, like a phone held at arm's length
        if a.dialogue_gain_db:
            chain.append(f"volume={a.dialogue_gain_db}dB")
        chain += ["adelay=250|250", f"apad=whole_dur={duration:.3f}", f"atrim=0:{duration:.3f}"]
        inputs += ["-i", str(voice)]
        chains.append(f"[{n}:a]aformat=sample_rates=44100:channel_layouts=mono,{','.join(chain)}[v]")
        labels.append("[v]")
        n += 1
    if a.room_tone:
        rt = room_tone_file(cfg, location)
        if rt:
            inputs += ["-stream_loop", "-1", "-i", str(rt)]
            chains.append(f"[{n}:a]aformat=sample_rates=44100:channel_layouts=mono,atrim=0:{duration:.3f},volume=0.6[r]")
        else:
            inputs += ["-f", "lavfi", "-i", f"anoisesrc=color=brown:amplitude={a.room_tone_level}:sample_rate=44100:d={duration:.3f}"]
            chains.append(f"[{n}:a]aformat=sample_rates=44100:channel_layouts=mono,lowpass=f=3000[r]")
        labels.append("[r]")
        n += 1
    if not labels:
        inputs += ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono:d={duration:.3f}"]
        chains.append("[0:a]anull[s]")
        labels.append("[s]")
    mix = f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest,atrim=0:{duration:.3f}[out]" \
        if len(labels) > 1 else f"{labels[0]}anull[out]"
    ff.run([ffm, "-y", "-hide_banner", *inputs, "-filter_complex", ";".join(chains + [mix]), "-map", "[out]",
            "-ac", "1", "-ar", "44100", str(out)])
    return out, notes
