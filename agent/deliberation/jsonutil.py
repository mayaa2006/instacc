"""Lenient JSON extraction from LLM output."""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str):
    """Return the first JSON object/array found in text, or raise ValueError."""
    if not text:
        raise ValueError("empty output")
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = cand.find(opener)
            while start != -1:
                depth, in_str, esc = 0, False, False
                for i in range(start, len(cand)):
                    ch = cand[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                        continue
                    if ch == '"':
                        in_str = True
                    elif ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(cand[start : i + 1])
                            except json.JSONDecodeError:
                                break
                start = cand.find(opener, start + 1)
    raise ValueError("no JSON found in output")


def normalize_options(obj) -> list[dict]:
    """Accept {"options":[...]}, [...], strings or dicts -> [{"text":..., "rationale":...}]."""
    if isinstance(obj, dict):
        for key in ("options", "candidates", "items", "results", "hooks", "ideas", "captions"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
        else:
            obj = [obj]
    out = []
    for it in obj if isinstance(obj, list) else []:
        if isinstance(it, str):
            out.append({"text": it.strip(), "rationale": ""})
        elif isinstance(it, dict):
            text = it.get("text") or it.get("option") or it.get("hook") or it.get("idea") or it.get("caption")
            if text:
                out.append({**it, "text": str(text).strip(), "rationale": str(it.get("rationale", "")).strip()})
    return [o for o in out if o["text"]]
