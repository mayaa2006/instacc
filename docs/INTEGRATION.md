# Agent ↔ video integration contract

The agent side lives in `agent/`, `shared/`, `config/`. The video side lives in `app/` (ComfyUI client, render scripts).
All endpoints are on `http://127.0.0.1:8000`. If `INSTACC_API_TOKEN` is set, send
`Authorization: Bearer <token>`.

## Who owns what
- **Agent side:** ideation, scripts (`VideoJob`), Persona Bible, continuity ledger, approvals, GPU
  scheduler, captions, posting, comments, and analytics.
- **Video side:** ComfyUI workflows, keyframes, character LoRA, clip chaining, ArcFace drift check,
  upscale and interpolation, and final mp4.

## Render loop (video side)
```
POST /api/gpu/mode {"mode":"render","holder":"video","ttl_s":3600}    # blocks until LLMs are off the GPU
GET  /api/render-queue                                                 # posts with status render_queued, each has .video_job
POST /api/posts/{id}/claim                                             # -> rendering
...render post.video_job...
POST /api/posts/{id}/rendered  {"media_path": "renders/<id>.mp4"}      # path absolute or relative to InstACC
  or POST /api/posts/{id}/render-failed {"error": "..."}
POST /api/gpu/mode {"mode":"think"}                                    # agent also calls ComfyUI /free
```
Re-send the render-mode call before the TTL runs out to keep the GPU. After `rendered`, the agent
writes captions, sends you the video on Telegram for approval, and schedules it.

## VideoJob (`shared/schemas/video_job.py`, DRAFT v0.1)
- `character_refs`: from `Persona.visual.reference_ids`. These are the keys for the character LoRA
  and reference images.
- `continuity`: outfit, location, lighting, props and time_of_day. Keep these identical across a
  series.
- `shots[]`: `duration_s` (2–5 s, sized for Wan 2.2 clips), `camera`, `action`, `setting`,
  `dialogue`, `on_screen_text`, `keyframe_prompt` (always set on shot 1), and
  `continue_from_previous` (seed from the previous clip's last frame).
- `seed_policy`: `locked_per_series` by default.
- `audio`: `voice_ref`, `music_mood`, `sfx`.
- `render`: **owned by the video side** (workflow, resolution, steps, fps). The agent never sets it.

The video side should propose changes to anything above by editing the schema in a commit, since
both halves import it.
