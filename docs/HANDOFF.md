# Handoff: agent side of instacc

Written at the end of the cloud session that built the agent side, for whoever continues on the PC.
Everything below is on GitHub branch **`ccr-9f4994da-q03f0a`** of `mayaa2006/instacc`.

## Latest changes (newest first)
- **Default is now FREE and fully local** (user request). `config/agent.yaml` has
  `slot_profile: local`: `alt` and `judge` both run on the local `main` Qwen 30B, so nothing goes to the
  cloud and nothing costs money.
  - To go back to the uncensored cloud models, set `slot_profile: cloud`; the profiles live under
    `slot_profiles:`.
  - **Variety with one model.** Everything is configurable per task in `config/tasks.yaml`, and the
    engine changes are in `agent/deliberation/engine.py`:
    1. Each role has its own sampling settings (`temperature`, `top_p`, `min_p`, `presence_penalty`) plus a
       fresh `seed` per call.
    2. `constraints: true`: each proposer gets a different random line from `creative_constraints`.
    3. `diversify: sequential`: GPU proposers run one by one and are told the "angles already taken". This is
       free on one GPU. CPU and cloud proposers run alongside.
    4. A `fast@1.2:wildcard` proposer (4B on CPU) was added to `hooks`, `idea` and `plan`.
    5. `dedupe: 0.9`: near-duplicate options are dropped using local embeddings (trace stage `dedupe`).
    6. `judge_panel: 2`: the judge scores as 2 random `judge_personas` on shuffled options, and the scores
       are averaged (`panel_scores` in results).
  - **New eval ablation `pipeline:no-variety`**, to measure whether the tricks above earn their time.
  - Tests: 55 pass. New ones are in `tests/test_variety.py`. The `settings` fixture applies the **cloud**
    profile so the routing tests keep meaning something; `local_settings` is the shipped config.
  - **To verify on the PC (no OpenRouter key needed):**
    1. Run one `hooks` task. Its trace should show only `local_*` endpoints, a `constraint:` note per
       proposer, a `dedupe` step, and 2 `judge` steps with `panel:` notes.
    2. Record the total time. Sequential proposers plus the panel judge make it slower than a single pass.
    3. Then run eval variants `single:main`, `pipeline` and `pipeline:no-variety` (limit 10), and let the
       user rate them blind.
  - **Verified live on llama-server (cloud, tiny model):** `top_p`, `min_p`, `top_k`, `presence_penalty` and
    `seed` are accepted together with JSON mode, and the same seed reproduces the same output.
  - **Not verified:** a full multi-proposer run end to end on real hardware. The cloud CPU was too slow
    (0.5 tok/s with two servers), so the PC run is the first real one.
  - **Possible issue:** at high temperature with JSON mode, a small model can ramble up to `max_tokens`. If the
    `fast` wildcard is slow or truncated on the PC, lower its temperature in `config/tasks.yaml` (roles →
    wildcard) or cap its output.
- **Cloud models are now uncensored / low-refusal** (user request; now the `cloud` profile, not the default). In `config/agent.yaml`:
  - `alt` → `cognitivecomputations/dolphin-mistral-24b-venice-edition`
  - `judge` → `nousresearch/hermes-4-405b`

  To verify on the PC:
  1. With `OPENROUTER_API_KEY` set, run one `idea` task and check its trace. The `alt` and `judge` steps
     should show the `openrouter` endpoint, with no `rerouted: refusal->local` notes on normal prompts.
  2. Check that the judge returns parseable JSON (scores appear in the result).
  3. If Hermes' reasoning makes the JSON unreliable, try a non-reasoning judge (e.g.
     `sao10k/l3.1-euryale-70b`, also on OpenRouter) and re-run the eval.

  Cost per call is lower than before, since Sonnet was $3/$15 per million tokens and these are $0.20–1 in
  and $0.90–3 out.

  Things that did not change:
  - Private task types still never use the cloud.
  - The refusal fallback to the local model stays on.
  - Embeddings are local only, enforced in code.

## 1. Current situation
- The PC folder `C:\Users\nehem\Downloads\InstACC` holds the video instance's work. It is
  **uncommitted, and the local repo has no commits yet**.
- None of the agent code has run on the PC yet. It was built and tested in a Linux cloud container.

### First job: bring the branch in without overwriting anything
1. `tools\git\cmd\git.exe status`: see what's there. Don't touch `comfyui/ python/ llm/ tools/ cache/`.
2. `tools\git\cmd\git.exe fetch origin ccr-9f4994da-q03f0a`
3. Check which branch files already exist locally:
   `tools\git\cmd\git.exe ls-tree -r --name-only FETCH_HEAD`
   - The likely clashes are `.gitignore` and maybe `README.md` or `docs/`.
   - Back up any clashing local files, check out the branch, then merge their contents back by hand.
   - The `.gitignore` must keep both sets of entries.
4. Commit the video side's own work on top. **Ask the user before pushing.**

## 2. What exists (agent side)

| Area | Where | Notes |
|---|---|---|
| Plan / design | `docs/AGENT_PLAN.md` | Why multi-model deliberation can beat one model; phases 1–5 |
| Setup guide | `docs/AGENT_SETUP.md` | Install, models, `.env`, Telegram, Android/Tailscale, publishing, evals |
| Video contract | `docs/INTEGRATION.md`, `shared/schemas/video_job.py` | Render-queue API, GPU mode API, VideoJob v0.1 (draft) |
| Config | `config/agent.yaml`, `config/tasks.yaml`, `config/llm-models.yaml` | Slots, servers, publishing, autonomy, deliberation per task |
| LLM client | `agent/models/client.py` | Slot routing; private tasks never go to the cloud; budget cap; refusal and cloud-error fallback to local |
| Deliberation | `agent/deliberation/` | Router; independent proposers; optional cross-critique; synthesizer; judge with loops; full trace |
| GPU scheduler | `agent/scheduler/` | Think/render modes, starts and stops `llama-server`, render lease TTL, ComfyUI `/free` on think |
| Memory | `agent/memory/` | Versioned Persona Bible, approved outputs with embeddings, continuity ledger (proposed → confirmed facts) |
| Content pipeline | `agent/content/` | Plan → shot list (VideoJob) → render queue → caption approval → scheduled → published; retry |
| Publishing | `agent/publish/` | Instagram Reels (FB Login resumable upload, `is_ai_generated`); TikTok inbox/direct (`is_aigc`, rotating refresh token) |
| Comments | `agent/community/comments.py` | IG only (TikTok API has none). Triage, drafts, injection guard, optional auto-reply for low-risk praise |
| Analytics | `agent/analytics/service.py` | Metrics at 1h/24h/7d, judge-vs-results rank correlation, memories tagged with real results |
| Autopilot | `agent/autopilot.py` | Publishes due posts, polls comments, collects metrics, keeps the TikTok token fresh |
| Evals | `agent/evals/harness.py`, `evals/golden.yaml` | Pipeline vs single model vs ablations; blind human votes plus a position-swapped LLM judge |
| Web app | `agent/api/static/index.html` (+ `classic.html`) | Phone + PC, installable PWA. Tabs: Inbox / Create / Calendar / Stats / More |
| Telegram | `agent/bots/telegram_bot.py` | Tasks, `/plan` multi-select, shot-list review, video + caption approval, comments, `/rate`, `/render`, `/think` |
| Launchers | `start-agent.bat`, `install-agent-deps.bat`, `get-llm-models.bat` | CRLF; use the portable `python\` |
| CI | `.github/workflows/tests.yml` | Ubuntu + Windows. Hasn't run yet: no default branch on GitHub |

## 3. What was verified, and how
- **55 tests pass** (originally 50) on Python 3.13 and 3.12, using fake LLMs and fake IG/TikTok APIs.
  - The fakes check request shapes: headers, `is_ai_generated`/`is_aigc`, chunk ranges and token rotation.
- **Live run against real `llama-server`, using a tiny Qwen3-0.6B on CPU in the cloud:**
  - the scheduler starts, stops and restarts servers across mode switches
  - the full propose → synthesize → judge flow works
  - cloud fallback works with no key
  - embeddings work with `--pooling mean`
  - the router picks task types
  - a valid VideoJob is produced
  - comment triage works
- **Web app:** checked in Playwright on emulated iPhone 13, Pixel 7 and a 1440 px desktop, in light and
  dark mode.
  - no JS errors and no horizontal overflow
  - the Android Back gesture works
  - Chrome reports no installability errors
- The IG/TikTok API details were checked against Meta/TikTok docs (Nov 2026, Graph API v27.0).
- **Not verified:**
  - anything on Windows or the real 3090 / 30B model
  - real IG/TikTok/Telegram calls
  - real Android or iPhone devices
  - `tailscale serve --bg 8000` syntax

## 4. Next steps (in order)
1. Merge the branch into the PC repo (section 1).
2. `install-agent-deps.bat`, then `python\python.exe -m pip install pytest pytest-asyncio`, then
   `python\python.exe -m pytest -q`. Fix anything Windows-specific.
3. Check disk space, **ask the user**, then `get-llm-models.bat`. It downloads into `llm\models\`:
   - main: Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated Q4_K_M, 18.6 GB
   - fast: the 4B version of the same, 2.5 GB
   - embed: nomic-embed v1.5 Q8
4. `start-agent.bat`. Confirm `main` loads fully on the GPU with 16k context and q8 KV.
   - Record VRAM, tokens/s and load time.
   - If it OOMs, reduce `-c` or use `--n-cpu-moe` (but RAM is only 16 GB).
   - Logs are in `data\logs\`.
5. User setup: `.env` (OpenRouter key, API token, Telegram token + user ID) and `config\persona.yaml`.
6. Run a small eval (`single:main, single:judge, pipeline`, limit 10) and have the user rate it on the
   phone.
7. Agree the VideoJob v0.1 with the video side once its first Wan renders work. Wire the video side to
   the render-queue API in `docs/INTEGRATION.md`.
8. Publishing stays disabled until the user has the Meta app (FB Login + System User token) and the
   TikTok app (audit pending, so use `mode: inbox`).

## 5. Open questions for the user
- **RAM:** 15.9 GB measured vs "64 GB" stated. If they upgrade, uncomment the `render:` profiles in
  `config/agent.yaml`.
- **Cloud model picks (changed at the user's request: uncensored cloud models):**
  - `alt` = `cognitivecomputations/dolphin-mistral-24b-venice-edition` (Mistral family)
  - `judge` = `nousresearch/hermes-4-405b` (Llama family)
  - Both IDs were checked against OpenRouter's live model list in Nov 2026. The cap is still $20/month.
  - Hermes 4 can emit `<think>` blocks; the client strips them, and the judge slot has
    `max_tokens: 3000` so reasoning doesn't truncate the JSON.
- **Tailscale:** it's a system install, so it breaks the portable rule. It's only needed for a full
  Android app install and access away from home. The user hasn't decided; Telegram works without it.

## 6. User context
- Works from Windows + an **Android** phone; wants simple UIs.
- Wants uncensored local models: abliterated GGUFs, with the cloud only for public, non-sensitive steps.
- Top video priorities: temporal and cross-video character consistency.
- Prefers being asked before big downloads and pushes.
