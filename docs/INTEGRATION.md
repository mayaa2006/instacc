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

## VideoJob (`shared/schemas/video_job.py`, DRAFT v0.2: phone realism)
Every video should look like the persona filmed it herself on **one phone**. The rules live in
`config/style.yaml` (active style `phone_ugc`), and the agent enforces them before a job reaches the render
queue.

**Job-level fields**
- `style`: `phone_ugc`.
- `negative_prompt`: feed it to the video model as-is. It covers cinematic looks, gimbals, drones, studio
  lighting, CGI and plastic skin.
- `character_refs`: from `Persona.visual.reference_ids`. These are the keys for the character LoRA and
  reference images.
- `continuity`: outfit, location, lighting, props and time_of_day. Keep these identical across a series.
- `realism_warnings`: issues the agent couldn't fix. Usually empty; the user saw any before approving.
- `seed_policy`: `locked_per_series` by default.
- `audio`: `voice_ref`, `music_mood`, `sfx`.
- `render`: **owned by the video side** (workflow, resolution, steps, fps). The agent never sets it.

**Per-shot fields**
- `setup`: how the single phone is held.

  | setup | Meaning | Face visible | Hands free |
  |---|---|---|---|
  | `selfie` | front camera at arm's length | yes | 1 |
  | `propped` | phone leaned on something, static | yes | 2 |
  | `pov` | back camera, her hands | no | 1 |
  | `mirror` | filming her reflection | yes | 1 |
  | `screen` | screen or text close-up | no | 1 |

- `render_prompt`: **use this as the video-model prompt.** It combines action, setting, appearance (only when
  her face is visible), outfit, lighting, camera motion, the setup's phone wording and the style's
  "shot on a smartphone…" look.
- `keyframe_prompt`: the first-frame image prompt, already with phone-photo wording. It's set on shot 1 and
  on any shot with `continue_from_previous: false`.
- `duration_s`: 2–5 s per take; 10–30 s per video; 2–6 shots.
- `camera_motion`, `dialogue`, `on_screen_text`.
- `continue_from_previous`: seed from the previous take's last frame. **A change of `setup` is a jump cut**,
  so don't chain frames across it: start that take from a new keyframe of the same scene.

### Realism advice for the video side (recommendations, not yet implemented)
1. **Keyframes:** generate the first frame as a *phone photo* (image model with the character LoRA, using
   `keyframe_prompt`), not a polished render. No beauty retouching, natural skin texture, slightly off-centre
   framing.
2. **Resolution and frame rate:** render 720×1280 (9:16), then interpolate to 30 fps. Phones record 30 fps;
   24 fps reads as "film".
3. **No grading:** keep colours flat and natural, with no LUTs, vignettes or film grain.
4. **Phone artefacts in post** (cheap with ffmpeg):
   - a little sensor noise
   - mild H.264 compression
   - tiny auto-exposure drift
   - for `selfie`/`pov` takes, a gentle handheld wobble if the model made the shot too steady
5. **Jump cuts:** hard cuts between takes, no transitions.
6. **Audio:** phone-mic sound sells realism. Use dialogue spoken to camera (TTS with `voice_ref`) plus quiet
   room tone. Avoid polished music beds unless the concept calls for them.
7. **Hands:** `negative_prompt` covers deformed hands. Hands are on screen in every `pov` shot, so check them
   in the ArcFace/drift pass too.

The video side should propose changes to anything above by editing the schema in a commit, since both halves
import it.
