# instacc: project memory for Claude Code

Read `docs/HANDOFF.md` first: it has the current state, what's verified, and the next steps.

## What this is
A self-hosted, semi-autonomous agent that plans, produces and posts content for an Instagram/TikTok
persona. It is controlled from the PC and an Android phone.

Two halves share this repo:
- **Agent side:** `agent/`, `shared/`, `config/`, `evals/`, `tests/`. It covers the LLM deliberation
  engine, GPU scheduler, posting, comments, analytics, Telegram bot and the web app.
- **Video side:** `app/` plus the portable installs (`comfyui/`, `llm/llama.cpp/`, `python/`, `tools/`).
  It runs Wan 2.2 in ComfyUI.

The contract between the halves is `docs/INTEGRATION.md` and `shared/schemas/`.

## Hard rules
- **Portable:** everything lives in `C:\Users\nehem\Downloads\InstACC`.
  - No system-wide installs, PATH or registry changes.
  - Caches go to `cache\` (PIP_CACHE_DIR, HF_HOME, TORCH_HOME).
  - Use `python\python.exe` (embeddable 3.12; its `._pth` has `..`, so `agent`, `shared` and `app` import)
    and `tools\git\cmd\git.exe`.
- Keep `.ps1` files ASCII-only (PowerShell 5.1) and `.bat` files CRLF (`.gitattributes` enforces it).
- Never commit `comfyui/ python/ tools/ llm/ cache/ logs/ data/ .env` or model weights.
- **Ask the user before:**
  - downloading big models (`get-llm-models.bat` is about 21 GB; the video models about 32 GB)
  - anything that posts publicly
- Official Instagram/TikTok APIs only: no browser bots. The AI-generated label is always on.

## Hardware (measured by scripts\hwcheck.ps1)
- RTX 3090 24 GB, Ryzen 7 5800X, **15.9 GB RAM**. The user once said 64 GB; this is unconfirmed, so trust
  the measurement.
- The GPU is either in think mode (LLMs) or render mode (Wan), never both. With 16 GB RAM the LLMs
  stop during renders.

## Dev commands
- Tests: `python\python.exe -m pytest -q` (83 tests; needs `pip install pytest pytest-asyncio`).
- Run: `start-agent.bat`. It listens on the network by default, prints a phone QR code and opens the PC
  browser signed in. The app is at `/`, the classic dashboard at `/classic`. Phones pair by QR
  (`agent/pairing.py`) or Telegram `/app`.
- Config:
  - `config/agent.yaml`: models, servers, publishing, autonomy.
  - `config/tasks.yaml`: deliberation per task type.
  - `config/style.yaml`: the filming style. Videos must look like ONE person filmed them on ONE phone; this
    file holds the phone setups, banned camera terms, look/negative prompt and timing, enforced in
    `agent/content/realism.py`.
- **Models are free and local by default** (`slot_profile: local`).
  - `slot_profile: cloud` switches `alt`/`judge` to uncensored OpenRouter models (paid, capped).
  - With one local model, variety comes from role sampling, seeds, creative constraints, sequential
    anti-repetition, the 4B wildcard, dedupe and the judge panel (all in `config/tasks.yaml`).

## Code conventions learned the hard way
- **Local LLM sampling must stay mild:** temperature <= 0.95 and **no presence/repetition penalties**. On
  llama.cpp, penalties hit JSON punctuation and produce nonsense. Thinking is switched off per request
  (`chat_template_kwargs`). Diagnose with `llm-doctor.bat`.
- FastAPI routes that start background work (`asyncio.create_task`, e.g. `tasks.create`,
  `posts.create`, `posts.retry`) **must be `async def`**. Sync routes run in a threadpool with no event loop.
- Every call to a local LLM goes through `GpuScheduler.lease(server)`. Calls to a server that is
  stopped in the current mode wait for think mode.
- Follower comments and other external text are untrusted. They are always delimited in prompts and
  never auto-acted on unless low risk; see `_SUSPICIOUS` in `agent/community/comments.py`.
- Test fakes live in `tests/conftest.py`: `FakeLLM` routes on prompt text, and `FakePlatforms` fakes the
  IG and TikTok APIs.
