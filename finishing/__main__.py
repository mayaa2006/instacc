"""CLI for the video side:  python -m finishing --job job.json --takes t1.mp4 t2.mp4 --out final.mp4"""
from __future__ import annotations

import argparse
import json
import sys

from finishing.config import load_finishing
from finishing.pipeline import finish


def main() -> int:
    ap = argparse.ArgumentParser(prog="finishing")
    ap.add_argument("--job", required=True, help="VideoJob JSON file")
    ap.add_argument("--takes", nargs="+", required=True, help="rendered takes, one per shot, in order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-on-bad-takes", action="store_true")
    a = ap.parse_args()
    job = json.loads(open(a.job, encoding="utf-8").read())
    r = finish(job, a.takes, a.out, load_finishing(), stop_on_bad_takes=a.stop_on_bad_takes)
    print(json.dumps(r.as_dict(), indent=2))
    return 0 if r.passed else 1


if __name__ == "__main__":
    sys.exit(main())
