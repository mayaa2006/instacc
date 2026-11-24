"""Cheap checks that catch broken model output before it reaches the user.

Typical failure modes of local models at high temperature: drifting into another script (Qwen -> Chinese),
symbol soup, glued-together "words", and repetition loops. These checks are deterministic and fast.
"""
from __future__ import annotations

import re

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_CYR = re.compile(r"[Ѐ-ӿ]")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def garbled_reason(text: str, expect_latin: bool = True) -> str | None:
    """Why `text` looks broken, or None if it looks like real writing."""
    t = (text or "").strip()
    if len(t) < 3:
        return "empty"
    if expect_latin and (_CJK.search(t) or len(_CYR.findall(t)) > 3):
        return "wrong language/script"
    body = re.sub(r"#\w+|@\w+|https?://\S+", "", t)          # hashtags/handles are fine
    letters = sum(ch.isalpha() for ch in body)
    if body and letters / max(1, len(body.replace(" ", ""))) < 0.6:
        return "mostly symbols"
    words = _WORD.findall(body)
    if words and max(len(w) for w in words) > 28:
        return "run-together words"
    if len(words) >= 8:
        lw = [w.lower() for w in words]
        if len(set(lw)) / len(lw) < 0.35:
            return "repetition loop"
        for n in (2, 3, 4):
            grams = [" ".join(lw[i:i + n]) for i in range(len(lw) - n + 1)]
            if grams and max(grams.count(g) for g in set(grams)) >= 4:
                return "repetition loop"
    return None


def persona_expects_latin(persona) -> bool:
    lang = (getattr(persona, "model_extra", None) or {}).get("language", "English") if persona else "English"
    return not re.search(r"chinese|japanese|korean|russian|ukrainian|mandarin|cantonese", str(lang), re.I)
