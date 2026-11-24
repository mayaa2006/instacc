"""Realism checks for shot lists: could ONE person film this with ONE phone?

Deterministic and fast; issues are fed back to the model for a rewrite, and anything left is shown to the
user on the shot-list card. Also builds each shot's render prompt for the video model.
"""
from __future__ import annotations

import re

from agent.config import FilmStyle

_TWO_HANDS = re.compile(r"\b(both hands|two hands|claps?|clapping|ties? (her|a) |braids?|kneads?|"
                        r"types? on (a|her) (laptop|keyboard)|pours? .* while (stirring|holding)|"
                        r"holds? .* (in each hand|with both))", re.I)
_FACE = re.compile(r"\b(her face|she smiles|smiles at|looks (into|at) the (camera|lens)|she talks to (the )?camera|"
                   r"winks|her eyes|facial expression|she grins|she laughs into)\b", re.I)
_OTHER_PERSON_FILMING = re.compile(r"\b(friend films|someone films|camera operator|cameraman|filmed by (a|her) "
                                   r"(friend|boyfriend|partner))\b", re.I)


def check_shots(shots: list[dict], style: FilmStyle) -> list[str]:
    issues: list[str] = []
    setups = style.setups
    t = style.timing
    if not t.min_shots <= len(shots) <= t.max_shots:
        issues.append(f"use {t.min_shots}-{t.max_shots} shots (got {len(shots)})")
    total = sum(float(sh.get("duration_s") or 0) for sh in shots)
    if not t.min_total_s <= total <= t.max_total_s:
        issues.append(f"the whole video should be {t.min_total_s:g}-{t.max_total_s:g} seconds (got {total:g})")
    for i, sh in enumerate(shots, 1):
        text = " ".join(str(sh.get(k, "")) for k in ("camera", "action", "setting", "keyframe_prompt", "camera_motion"))
        low = text.lower()
        sid = (sh.get("setup") or "").strip().lower()
        if sid not in setups:
            issues.append(f"shot {i}: 'setup' must be one of {', '.join(setups)} (got '{sid or 'nothing'}')")
            continue
        su = setups[sid]
        d = float(sh.get("duration_s") or 0)
        if not t.min_shot_s <= d <= t.max_shot_s:
            issues.append(f"shot {i}: each take should be {t.min_shot_s:g}-{t.max_shot_s:g} seconds (got {d:g})")
        for term in style.banned_terms:
            if term.lower() in low:
                issues.append(f"shot {i}: '{term}' can't be filmed by one person with one phone")
        if not su.face_visible and _FACE.search(text):
            issues.append(f"shot {i}: in a '{sid}' shot her face isn't visible; use 'selfie' or 'propped', "
                          "or describe only her hands and what she sees")
        if su.hands_free < 2 and _TWO_HANDS.search(text):
            issues.append(f"shot {i}: she needs both hands, but one is holding the phone; use 'propped'")
        if _OTHER_PERSON_FILMING.search(text):
            issues.append(f"shot {i}: nobody else holds the camera")
    return issues


def render_prompt(shot: dict, continuity: dict, style: FilmStyle, appearance: str = "") -> str:
    """Video-model prompt: subject + action + place + continuity + how the phone is held + phone look."""
    su = style.setups.get((shot.get("setup") or "").lower())
    parts = [
        shot.get("action", "").strip().rstrip("."),
        shot.get("setting", "").strip().rstrip("."),
    ]
    if su and su.face_visible and appearance:
        parts.append(f"the woman: {appearance.strip().rstrip('.')}")
    c = continuity or {}
    if c.get("outfit"):
        parts.append(f"wearing {c['outfit']}")
    if c.get("location") and c["location"].lower() not in (shot.get("setting") or "").lower():
        parts.append(c["location"])
    if c.get("lighting"):
        parts.append(c["lighting"])
    if shot.get("camera_motion"):
        parts.append(shot["camera_motion"])
    if su:
        parts.append(su.prompt)
    if style.look:
        parts.append(style.look.strip())
    return ". ".join(p for p in parts if p) + "."


def auto_repair(shots: list[dict], style: FilmStyle) -> tuple[list[dict], list[str]]:
    """Fix the mechanical violations without another model call: pick the phone setup the shot actually
    needs. Banned camera moves and timing problems are left for the model to rewrite."""
    fixed, notes = [], []
    setups = style.setups
    for i, sh in enumerate(shots, 1):
        sh = dict(sh)
        text = " ".join(str(sh.get(k, "")) for k in ("camera", "action", "setting", "keyframe_prompt"))
        sid = (sh.get("setup") or "").strip().lower()
        needs_face, needs_two = bool(_FACE.search(text)), bool(_TWO_HANDS.search(text))
        new = sid
        if sid not in setups:
            new = "propped" if needs_two else "selfie" if needs_face else "pov"
        elif needs_two and setups[sid].hands_free < 2:
            new = "propped"
        elif needs_face and not setups[sid].face_visible:
            new = "propped" if needs_two else "selfie"
        if new != sid and new in setups:
            notes.append(f"shot {i}: phone setup {sid or 'missing'} → {new}")
            sh["setup"] = new
        fixed.append(sh)
    # timing: clamp each take, then stretch or shrink takes (within limits) to hit the total length
    t = style.timing
    durs = [float(x.get("duration_s") or t.min_shot_s) for x in fixed]
    new = [min(max(d, t.min_shot_s), t.max_shot_s) for d in durs]
    total = sum(new)
    if total < t.min_total_s:
        room = sum(t.max_shot_s - d for d in new)
        if room > 0:
            need = min(t.min_total_s - total, room)
            new = [d + (t.max_shot_s - d) / room * need for d in new]
    elif total > t.max_total_s:
        room = sum(d - t.min_shot_s for d in new)
        if room > 0:
            cut = min(total - t.max_total_s, room)
            new = [d - (d - t.min_shot_s) / room * cut for d in new]
    new = [round(d * 2) / 2 for d in new]   # half-second steps
    if new != durs:
        notes.append(f"take lengths {'/'.join(f'{d:g}' for d in durs)} s → {'/'.join(f'{d:g}' for d in new)} s")
        for x, d in zip(fixed, new):
            x["duration_s"] = d
    return fixed, notes
