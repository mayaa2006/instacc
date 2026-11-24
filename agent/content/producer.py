"""Producer: turns a post concept into a validated VideoJob (shot list) using the local model."""
from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from agent.content.realism import auto_repair, check_shots, render_prompt
from agent.deliberation.jsonutil import extract_json
from shared.schemas import VideoJob

log = logging.getLogger(__name__)

PROMPT = """You are directing a short video that must look like a REAL phone video filmed by the persona herself.
Write the shot list for ONE vertical (9:16) video.

{persona}

{continuity}

SERIES: {series}
CONCEPT: {concept}
{note}
FILMING STYLE: {style_name}
{style_brief}

HOW THE ONE PHONE CAN BE HELD (each shot uses exactly one "setup" id):
{setups}

RULES:
{rules}
- Each shot is one continuous take of {min_shot_s:g}-{max_shot_s:g} seconds (the video model makes ~5 s clips).
  Use {min_shots}-{max_shots} shots, {min_total_s:g}-{max_total_s:g} seconds in total.
- Changing setup between shots is a jump cut: she stopped recording and moved the phone. That's normal.
- Describe only what a phone would actually record: concrete, physical, everyday. Keep outfit, place and
  light identical across shots unless the concept needs a change. Respect the continuity ledger exactly.
- Shot 1 hooks in the first second (a surprising action, or a line said straight into the camera) and has a
  keyframe_prompt: a detailed description of its first frame as the phone would capture it.
- "camera_motion" is how the phone moves: e.g. "static", "slight handheld shake", "she tilts the phone down to the cup".
{feedback}
Reply with ONLY JSON in this shape:
{{"title": "...", "concept": "...",
  "continuity": {{"outfit": "...", "location": "...", "lighting": "...", "props": ["..."], "time_of_day": "..."}},
  "shots": [{{"setup": "selfie|propped|pov|...", "duration_s": 4, "camera": "framing as the phone sees it",
             "camera_motion": "...", "action": "...", "setting": "...", "dialogue": "", "on_screen_text": "",
             "keyframe_prompt": "...", "continue_from_previous": true}}],
  "audio": {{"music_mood": "...", "sfx": ["..."]}}}}"""

# used when no config/style.yaml exists
LEGACY_RULES = "- Every shot must be visually concrete: who, where, what moves, camera framing and movement."


class Producer:
    def __init__(self, db, llm, memory, continuity, style=None, style_id: str | None = None):
        self.db = db
        self.llm = llm
        self.memory = memory
        self.continuity = continuity
        self.style = style          # FilmStyle from config/style.yaml (None = no realism rules)
        self.style_id = style_id

    async def video_job(self, post: dict, note: str | None = None) -> VideoJob:
        persona = self.memory.persona()
        series = self.continuity.get_series(post["series_id"]) if post.get("series_id") else None
        style = self.style
        user = PROMPT.format(
            persona=persona.to_prompt() if persona else "",
            continuity=self.continuity.to_prompt(post.get("series_id")),
            series=f"{series['name']}: {series.get('description') or ''}" if series else "(standalone video)",
            concept=post["concept"],
            note=f"OWNER'S NOTES: {note}\n" if note else "",
            style_name=style.name if style else "natural",
            style_brief=style.brief if style else "",
            setups="\n".join(f"- {k}: {v.description}" for k, v in style.setups.items()) if style else
                   "- (any)",
            rules="\n".join(f"- {r}" for r in style.rules) if style else LEGACY_RULES,
            feedback="",
            **(style.timing.model_dump() if style else
               {"min_shots": 2, "max_shots": 6, "min_shot_s": 2, "max_shot_s": 5, "min_total_s": 10, "max_total_s": 30}),
        )
        messages = [{"role": "user", "content": user}]
        last_err = None
        warnings: list[str] = []
        for attempt in range(4):
            res = await self.llm.chat("main", messages, temperature=0.5, sensitivity="private",
                                      json_mode=True, max_tokens=2500)
            self.db.trace_add(f"post:{post['id']}", "produce", loop=attempt, slot=res.slot, model=res.model,
                              endpoint=res.endpoint, output=res.text, latency_ms=res.latency_ms)
            try:
                obj = extract_json(res.text)
                if not isinstance(obj, dict):
                    raise ValueError("expected a JSON object")
                ledger = self.continuity.for_video_job(post.get("series_id"))
                cont = {**ledger, **{k: v for k, v in (obj.get("continuity") or {}).items() if v}}
                job = VideoJob.model_validate({
                    **obj, "continuity": cont, "post_id": post["id"], "series_id": post.get("series_id"),
                    "character_refs": persona.visual.reference_ids if persona else [],
                    "audio": {**(obj.get("audio") or {}),
                              "voice_ref": (persona.model_extra or {}).get("voice_ref") if persona else None},
                })
            except (ValueError, ValidationError) as e:
                last_err = e
                messages += [{"role": "assistant", "content": res.text},
                             {"role": "user", "content": f"That was invalid: {str(e)[:800]}\nReply again with ONLY the corrected JSON."}]
                continue
            if style:
                shots = [sh.model_dump() for sh in job.shots]
                issues = check_shots(shots, style)
                if issues:   # fix what's mechanical first: no extra model call needed
                    repaired, notes = auto_repair(shots, style)
                    left = check_shots(repaired, style)
                    if notes and len(left) < len(issues):
                        self.db.trace_add(f"post:{post['id']}", "realism", loop=attempt,
                                          note=f"auto-fixed {len(notes)} thing(s)", output="\n".join(notes))
                        job = job.model_copy(update={"shots": [type(job.shots[0]).model_validate(x) for x in repaired]})
                        issues = left
                if issues and attempt < 3:
                    warnings = issues
                    self.db.trace_add(f"post:{post['id']}", "realism", loop=attempt,
                                      note=f"{len(issues)} issue(s), asking for a rewrite", output="\n".join(issues))
                    messages += [{"role": "assistant", "content": res.text},
                                 {"role": "user", "content": "This can't be filmed by one person with one phone:\n"
                                  + "\n".join(f"- {i}" for i in issues)
                                  + "\nFix these shots and reply again with ONLY the corrected JSON."}]
                    continue
                return self._finish(job, style, persona, issues)
            return job
        raise RuntimeError(f"could not produce a valid VideoJob: {last_err}")

    def _finish(self, job: VideoJob, style, persona, issues: list[str]) -> VideoJob:
        """Attach the video-model prompts and anything the checker couldn't get fixed."""
        appearance = persona.visual.appearance if persona else ""
        cont = job.continuity.model_dump()
        for sh in job.shots:
            sh.render_prompt = render_prompt(sh.model_dump(), cont, style, appearance)
            su = style.setups.get((sh.setup or "").lower())
            if sh.keyframe_prompt and su:
                sh.keyframe_prompt = f"{sh.keyframe_prompt.rstrip('.')}. {su.prompt}. {style.look.strip()}"
        job.style = self.style_id or ""
        job.negative_prompt = " ".join(style.negative_prompt.split())
        job.realism_warnings = issues
        return job


SETUP_ICON = {"selfie": "🤳", "propped": "📱", "pov": "👀", "mirror": "🪞", "screen": "🖥"}


def summarize_job(job: dict) -> str:
    lines = [f"🎬 {job.get('title') or 'video'} · {sum(s.get('duration_s', 0) for s in job.get('shots', [])):.0f}-second video"]
    c = job.get("continuity") or {}
    bits = [c.get(k) for k in ("outfit", "location", "lighting") if c.get(k)]
    if bits:
        lines.append("👗 " + " · ".join(bits))
    for i, s in enumerate(job.get("shots", []), 1):
        say = f' — "{s["dialogue"]}"' if s.get("dialogue") else ""
        icon = SETUP_ICON.get(s.get("setup", ""), "")
        lines.append(f"{i}. {icon} ({s.get('duration_s')}s, {s.get('setup') or s.get('camera')}) {s.get('action')}{say}")
    for w in job.get("realism_warnings") or []:
        lines.append(f"⚠️ {w}")
    return "\n".join(lines)


def job_json(job: VideoJob) -> str:
    return json.dumps(job.model_dump(mode="json"), ensure_ascii=False)
