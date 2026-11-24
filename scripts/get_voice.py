"""Install the free Piper voice engine into the portable Python and download one voice (~63 MB).

Usage: get-voice.bat [voice]   e.g.  en_US-amy-medium (default), en_GB-jenny_dioco-medium, en_US-lessac-medium
Browse voices: https://huggingface.co/rhasspy/piper-voices
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def main() -> int:
    voice = sys.argv[1] if len(sys.argv) > 1 else "en_US-amy-medium"
    lang, name, quality = voice.split("_", 1)[0], voice.split("-")[1], voice.split("-")[-1]
    region = voice.split("-")[0]
    print("Installing piper-tts into the portable Python...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-warn-script-location", "piper-tts"], check=True)
    dest = ROOT / "voices"
    dest.mkdir(exist_ok=True)
    for suffix in (".onnx", ".onnx.json"):
        url = f"{BASE}/{lang}/{region}/{name}/{quality}/{voice}{suffix}"
        out = dest / f"{voice}{suffix}"
        if out.exists() and out.stat().st_size > 1000:
            print(f"  already have {out.name}")
            continue
        print(f"  downloading {out.name}")
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
    print(f"\nDone. In config/finishing.yaml set:\n  voice:\n    engine: piper\n    piper_voice: voices/{voice}.onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
