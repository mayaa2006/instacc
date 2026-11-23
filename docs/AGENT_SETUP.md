# Agent setup (Phase 1)

Everything stays inside the `InstACC` folder (portable, no system installs), alongside the video
side's `comfyui\`, `python\`, `llm\llama.cpp\` and `tools\`.

## 1. Install the agent's Python packages
```
install-agent-deps.bat
```
This installs `requirements-agent.txt` into the portable `python\`. FastAPI, uvicorn and pydantic are
already there; this adds httpx, pyyaml and python-telegram-bot.

## 2. Download the LLM weights (~21 GB)
```
get-llm-models.bat --list     # show what will be downloaded
get-llm-models.bat            # resumable; run again if interrupted
```
The files land in `llm\models\`:

| Slot | Model | Size | Runs on |
|---|---|---|---|
| main | Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated Q4_K_M | 18.6 GB | GPU (think mode only) |
| fast | Huihui-Qwen3-4B-Instruct-2507-abliterated Q4_K_M | 2.5 GB | CPU (think mode only) |
| embed | nomic-embed-text-v1.5 Q8_0 | 0.14 GB | CPU (always) |

Check disk space first: the video models take ~32 GB and C: had ~81 GB free.

## 3. Secrets
Copy `.env.example` to `.env` and fill in:
- `OPENROUTER_API_KEY`: from openrouter.ai. Set a credit limit there as well as `budget.monthly_usd`
  in `config/agent.yaml`.
- `INSTACC_API_TOKEN`: any long random string. The dashboard asks for it once.
- `TELEGRAM_BOT_TOKEN`: create a bot with @BotFather.
- `TELEGRAM_ALLOWED_USERS`: your numeric Telegram ID (from @userinfobot). Everyone else is ignored.

## 4. Persona
Copy `config\persona.example.yaml` to `config\persona.yaml` and write your character before the
first start. After that, edit it in the dashboard (Persona tab). Every save is a new version.

## 5. Run
```
start-agent.bat                  # PC only: http://127.0.0.1:8000
start-agent.bat --host 0.0.0.0   # reachable from your phone on the same Wi-Fi / Tailscale
```
On start, the agent launches the three `llama-server` processes itself. Their logs go to
`data\logs\`. First load of `main` takes a minute or so.

The first time you bind to `0.0.0.0`, Windows Firewall asks about `python.exe`. Allow it on
**Private** networks only.

### Phone access
- **Telegram (fully portable, recommended):** the bot polls outward, so nothing is exposed.
  - `/hooks`, `/caption`, `/idea` or `/quick` followed by your request; or just send text and pick a type.
  - Results arrive with ✅ 1–5 / 🔁 Rethink / ❌ Reject / 📜 Trace buttons.
  - `/render 6` reserves the GPU for video for 6 h; `/think` hands it back.
- **Dashboard:** open `http://<pc-ip>:8000` on your phone, and use "Add to Home Screen" to get an app icon.
  Away from home you need Tailscale. Note that Tailscale is a **system install** (Windows service),
  which is the one exception to the portable rule. Skip it if you want to stay strictly portable;
  Telegram covers remote control.

## GPU modes (video integration)
The 3090 has 24 GB of VRAM, but there is only **16 GB of system RAM**, so the LLMs and Wan never share it:

| | think | render |
|---|---|---|
| main (GPU) | running | **stopped** |
| fast (CPU) | running | **stopped** (leaves RAM to ComfyUI) |
| embed (CPU) | running | running |
| Private tasks | run | **wait** until think mode |
| Public tasks | run | cloud slots run; local steps wait |

The video side should switch modes over HTTP:
```
POST /api/gpu/mode  {"mode": "render", "holder": "video", "ttl_s": 3600}   # blocks until LLMs are off the GPU
...render...
POST /api/gpu/mode  {"mode": "think"}                                        # also tells ComfyUI to /free its VRAM
```
Re-send the render call to extend the lease. If the video side crashes, the lease expires and
the agent takes the GPU back.

With a RAM upgrade to 64 GB, uncomment the `render:` profiles in `config/agent.yaml`. The models then
keep running on CPU during renders.

## Tests (dev)
```
python\python.exe -m pip install pytest pytest-asyncio
python\python.exe -m pytest
```

## Evals: is the "thinking" worth it? (Phase 2)
`evals/golden.yaml` has 30 sample requests (hooks, captions, ideas). **Rewrite them for your persona.**

Dashboard → **Eval** → enter variants → **Start eval**. Variants:
- `pipeline`: the task type's full config
- `pipeline:no-critique`, `pipeline:no-judge`, `pipeline:no-synth`, `pipeline:no-loops`: ablations (combine with commas)
- `single:main`, `single:judge`, `single:alt`: one model, one answer, no deliberation

Each case's top option per variant is compared in pairs:
- **You:** Eval → Rate (or `/rate` in Telegram) shows two unlabeled answers in random order.
- **LLM judge:** judges every pair twice with the order swapped. If the two verdicts disagree, the pair counts as a tie, which cancels the judge's position bias.

The report shows win rate, W/L/T, head-to-head, cost and time per case, filterable by you vs the LLM.
**Trust your own votes over the LLM judge's.** Drop any stage whose ablation doesn't lose.

A full run of 30 cases × 3 variants takes hours on the 3090. Run it in think mode, e.g. with `limit` 10 first.
