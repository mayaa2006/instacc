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

### The app (phone + PC)
Open `http://<pc-ip>:8000` and type the password (`INSTACC_API_TOKEN`). It's one page that adapts:
bottom tabs on a phone, a sidebar on a PC.
- **Inbox:** everything that needs you, one card each:
  - pick a caption for a finished video (plays inline)
  - approve a shot list
  - tick the week's ideas
  - pick hooks
  - send or skip comment replies
  - confirm story facts
  - retry anything that failed
- **Create:** "Plan my week", or ask for hooks, captions or video ideas. You can also add a post directly.
- **Calendar:** posts by day; tap one to watch, move, post now or drop.
- **Stats:** views, the best posts, whether the AI judge is right, and cloud spend.
- **More:** your character, story facts, accounts, GPU mode, and a link to the **classic** view
  (`/classic`) with evals, traces and raw settings.

### Android
**Quick (home Wi-Fi):** start with `start-agent.bat --host 0.0.0.0`, then open `http://<pc-ip>:8000` in Chrome
on the phone and use **⋮ → Add to Home screen**. You get an icon, but over plain `http` it opens inside a Chrome tab.
Find the PC's IP with `ipconfig` (IPv4 Address).

**Proper app (own window, works away from home):** Chrome only *installs* web apps over HTTPS. The easiest
HTTPS is Tailscale. This is the one system install; skip it if you want to stay strictly portable.
1. Install Tailscale on the PC and on the phone (Play Store), and sign in to both with the same account.
2. In the Tailscale admin console → DNS, enable **MagicDNS** and **HTTPS Certificates**.
3. On the PC: `tailscale serve --bg 8000`. This puts the agent at `https://<pc-name>.<tailnet>.ts.net`
   with a real certificate. Keep the agent on its default `127.0.0.1`, since Tailscale forwards to it.
4. Open that address in Chrome on the phone, then **⋮ → Install app**. It opens full-screen like any app,
   and long-pressing the icon gives "Inbox" and "Plan my week" shortcuts.

If you'd rather serve HTTPS directly, `tailscale cert <pc-name>.<tailnet>.ts.net` writes a certificate, and
`start-agent.bat --host 0.0.0.0 --ssl-certfile <file>.crt --ssl-keyfile <file>.key` serves it.

**Telegram** works on Android from anywhere with no setup beyond the bot, and pairs well with the app:
the bot pings you, and the app is for browsing and longer edits.

On an iPhone, use Safari → Share → "Add to Home Screen".

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

## Publishing (Phase 3)
Nothing is posted until you enable a platform in `config/agent.yaml` → `publishing`, and even then
every post needs your caption approval (`autonomy.publish: ask`).

### Instagram (recommended: Facebook Login)
1. Switch the Instagram account to **Professional** and link it to a Facebook Page.
2. At developers.facebook.com create an app and add **Instagram** (Facebook Login for Business).
   Request `instagram_basic`, `instagram_content_publish`, `instagram_manage_comments` and
   `instagram_manage_insights`. App Review is needed before it works for real.
3. In Business Settings, create a **System User**, give it the Page + Instagram account and the app,
   and generate a token. System User tokens don't expire. Put it in `.env` as `IG_ACCESS_TOKEN`, and
   your Instagram business account ID as `IG_USER_ID`.
4. Set `publishing.instagram.enabled: true`.

Why Facebook Login: only it supports uploading the video file straight from your PC. With
`login: instagram` (Business Login for Instagram) Meta must download the video from a **public URL**,
so you'd need to host the files somewhere (`public_media_base_url`).
The AI label is set with the official `is_ai_generated` flag.
The API allows 100 posts per 24 h.

### TikTok
1. At developers.tiktok.com create an app, add **Content Posting API** and **Login Kit**, and request
   `video.upload`, `video.publish`, `video.list` and `user.info.basic`.
2. Register a redirect URI and copy it to `publishing.tiktok.redirect_uri`. Put the client key/secret
   in `.env`.
   - If the URI points at this agent (`…/auth/tiktok/callback`), connecting is one click.
   - If it points anywhere else, copy the `code=` value from the address bar after logging in and paste
     it into Accounts.
3. Set `publishing.tiktok.enabled: true`, then go to System → Accounts → **Connect TikTok**.
   The agent refreshes the 2-hour access token itself. The refresh token lasts 180 days and rotates.

Until TikTok audits your app, direct posts are **private only**. That's why the default is
`mode: inbox`: the video lands in your TikTok drafts with the AI label off, so you add the caption
and the AI-generated label and post from the app. After the audit, set `mode: direct` and
`privacy_level: PUBLIC_TO_EVERYONE`. The `is_aigc` label is then set automatically.

TikTok's API doesn't expose comments, so only Instagram comments are triaged.

### Comments
- Every 15 min (`autopilot.comment_poll_min`) new Instagram comments are pulled for posts from the
  last 14 days. Spam, hostile comments and ones that need no answer are ignored. The rest get a
  drafted reply sent to Telegram (✅ / ✏️ / 🙈).
- Comments are untrusted text. They're never treated as instructions. Replies containing links,
  contact handles or similar are flagged high risk, as are comments containing injection or
  credential phrases.
- With `autonomy.comment_replies: auto_low_risk`, only low-risk praise is answered without you.

### Analytics
- Metrics are pulled 1 h, 24 h and 7 d after each post.
- The Analytics tab ranks posts and shows whether the judge's scores predicted views and engagement,
  using rank correlation. It needs 8+ posts before giving a verdict.
- Results are written back into memory ("top 20%, 12k views"), so future prompts imitate what
  actually worked.
