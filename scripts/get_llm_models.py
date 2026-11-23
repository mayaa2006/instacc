"""Download the agent's GGUF models into llm/models/ (resumable, no extra deps).

Usage:  python\\python.exe scripts\\get_llm_models.py [--only main,fast] [--list]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "llm" / "models"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(repo: str, file: str) -> None:
    url = f"https://huggingface.co/{repo}/resolve/main/{file}"
    out = DEST / file
    part = out.with_suffix(out.suffix + ".part")
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, read=300)) as c:
        total = int(c.head(url).headers.get("content-length", 0))
        if out.exists() and (not total or out.stat().st_size == total):
            print(f"  already have {file}")
            return
        for attempt in range(1, 6):
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            try:
                with c.stream("GET", url, headers=headers) as r:
                    if r.status_code == 416:  # already complete
                        break
                    r.raise_for_status()
                    mode = "ab" if r.status_code == 206 else "wb"
                    done = have if mode == "ab" else 0
                    t0, last = time.time(), 0.0
                    with open(part, mode) as f:
                        for chunk in r.iter_bytes(1 << 20):
                            f.write(chunk)
                            done += len(chunk)
                            if time.time() - last > 2:
                                last = time.time()
                                rate = (done - have) / max(1e-6, last - t0)
                                pct = f"{100 * done / total:5.1f}%" if total else ""
                                print(f"\r  {file}: {pct} {human(done)} / {human(total)}  {human(rate)}/s   ",
                                      end="", flush=True)
                break
            except httpx.HTTPError as e:
                print(f"\n  attempt {attempt} failed: {e}; retrying in {5 * attempt}s")
                time.sleep(5 * attempt)
        else:
            sys.exit(f"giving up on {file}; run again to resume")
    if total and part.stat().st_size != total:
        sys.exit(f"size mismatch for {file}; run again to resume")
    part.replace(out)
    print(f"\n  saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated names from config/llm-models.yaml")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    models = yaml.safe_load((ROOT / "config" / "llm-models.yaml").read_text(encoding="utf-8"))["models"]
    if args.only:
        wanted = set(args.only.split(","))
        models = [m for m in models if m["name"] in wanted]
    if args.list:
        for m in models:
            print(f"{m['name']:6} {m['repo']}/{m['file']}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    for m in models:
        print(f"[{m['name']}] {m['repo']}")
        download(m["repo"], m["file"])
    print("done.")


if __name__ == "__main__":
    main()
