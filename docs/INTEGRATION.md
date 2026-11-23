# Agent ↔ video integration contract

The agent side lives in `agent/`, `shared/`, `config/`. The video side lives in `app/` (ComfyUI client, render scripts).

## Who owns what
- **Agent side:** the "thinking" stage (ideation, critique, scripting, director shot list), the Persona Bible,
  the continuity ledger, the approval queue, the GPU scheduler, posting, and analytics.
- **Video side:** ComfyUI workflows, keyframes, character LoRA, clip chaining, ArcFace drift check,
  upscale and interpolation.

## GPU
Before loading Wan, call `POST http://127.0.0.1:8000/api/gpu/mode` with
`{"mode":"render","holder":"video","ttl_s":3600}`. The call returns only after the LLMs are off the GPU.
When done, call it with `{"mode":"think"}`. The agent then calls ComfyUI `POST /free` itself.
Send `Authorization: Bearer <INSTACC_API_TOKEN>` if a token is set.

## Shared schemas (`shared/schemas/`)
- `Persona`: exists. The `visual.reference_ids` field is how the video side finds the character LoRA and
  reference images. Read it from `GET /api/persona`.
- `VideoJob`: **to be defined together in Phase 3.** The proposed shape is in `docs/AGENT_PLAN.md` §4.
  The video side should own the render-specific fields (workflow, steps, resolution); the agent fills in
  the series, character, continuity and shots.
