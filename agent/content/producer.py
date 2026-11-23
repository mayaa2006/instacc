"""Producer: turns a post concept into a validated VideoJob (shot list) using the local model."""
from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from agent.deliberation.jsonutil import extract_json
from shared.schemas import VideoJob

log = logging.getLogger(__name__)

PROMPT = """You are the director for a recurring AI-generated character on Instagram Reels / TikTok.
Write the shot list for ONE vertical (9:16) video.

{persona}

{continuity}

SERIES: {series}
CONCEPT: {concept}
{note}
Hard constraints (the video model generates ~5 s clips, each seeded from the previous clip's last frame):
- 2 to 6 shots, each 2-5 seconds, 10-30 seconds total.
- Every shot must be visually concrete: who, where, what moves, camera framing and movement.
- Keep outfit, location and lighting identical across shots unless the concept requires a change.
- Respect the continuity ledger exactly. Never contradict it.
- The first shot needs a keyframe_prompt: a detailed still-image description of the first frame.
- Hook in shot 1: something visually surprising or a line that stops the scroll.

Reply with ONLY JSON matching this shape:
{{"title": "...", "concept": "...",
  "continuity": {{"outfit": "...", "location": "...", "lighting": "...", "props": ["..."], "time_of_day": "..."}},
  "shots": [{{"duration_s": 4, "camera": "...", "action": "...", "setting": "...", "dialogue": "",
             "on_screen_text": "", "keyframe_prompt": "...", "continue_from_previous": true}}],
  "audio": {{"music_mood": "...", "sfx": ["..."]}}}}"""


class Producer:
    def __init__(self, db, llm, memory, continuity):
        self.db = db
        self.llm = llm
        self.memory = memory
        self.continuity = continuity

    async def video_job(self, post: dict, note: str | None = None) -> VideoJob:
        persona = self.memory.persona()
        series = self.continuity.get_series(post["series_id"]) if post.get("series_id") else None
        user = PROMPT.format(
            persona=persona.to_prompt() if persona else "",
            continuity=self.continuity.to_prompt(post.get("series_id")),
            series=f"{series['name']}: {series.get('description') or ''}" if series else "(standalone video)",
            concept=post["concept"],
            note=f"OWNER'S NOTES: {note}\n" if note else "",
        )
        messages = [{"role": "user", "content": user}]
        last_err = None
        for attempt in range(3):
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
                return job
            except (ValueError, ValidationError) as e:
                last_err = e
                messages += [{"role": "assistant", "content": res.text},
                             {"role": "user", "content": f"That was invalid: {str(e)[:800]}\nReply again with ONLY the corrected JSON."}]
        raise RuntimeError(f"could not produce a valid VideoJob: {last_err}")


def summarize_job(job: dict) -> str:
    lines = [f"🎬 {job.get('title') or 'video'} · {sum(s.get('duration_s', 0) for s in job.get('shots', [])):.0f}s"]
    c = job.get("continuity") or {}
    bits = [c.get(k) for k in ("outfit", "location", "lighting") if c.get(k)]
    if bits:
        lines.append("👗 " + " · ".join(bits))
    for i, s in enumerate(job.get("shots", []), 1):
        say = f' — "{s["dialogue"]}"' if s.get("dialogue") else ""
        lines.append(f"{i}. ({s.get('duration_s')}s, {s.get('camera')}) {s.get('action')}{say}")
    return "\n".join(lines)


def job_json(job: VideoJob) -> str:
    return json.dumps(job.model_dump(mode="json"), ensure_ascii=False)
