# instacc — LLM Agent Plan

Plan for the agent half of instacc: a semi-autonomous agent hosted on your PC, controlled from PC or phone, that plans content, drives the video pipeline, and manages the Instagram/TikTok accounts. Its core is a multi-model **deliberation engine** that "thinks" better than any single local model.

---

## 1. The honest premise: how a swarm of local models beats one big private model

A single open-weight model is generally weaker than the best frontier private model. You don't close that gap by having models chat with each other for longer. Debate on its own often *amplifies* shared mistakes. The gap closes through four levers, roughly in order of payoff:

| Lever | Why it works | Where it lives in this design |
|---|---|---|
| **Grounding** | Real data such as your analytics, past posts, trend data and comments beats any amount of reasoning from priors. A private chatbot doesn't have *your* account history. | Context Builder + Memory |
| **Diversity of errors** | Models from different families (Qwen, Llama, Mistral, GLM, DeepSeek…) make *different* mistakes, so aggregating them cancels noise. Five copies of one model mostly agree with themselves. | Divergent Proposers |
| **Test-time compute with selection** | Generate N candidates, then *select* using a rubric or verifier. Selection is easier than generation, so a judge can pick better than it could write. | Critique → Synthesis → Judge |
| **Feedback learning** | Real engagement numbers tell you which outputs were actually good. Over time the judge and prompts calibrate to *your* audience, which no general model can do. | Outcome Loop |

Published support: Mixture-of-Agents (layered open models beat GPT-4o on AlpacaEval in 2024), self-consistency, best-of-N with verifiers, Reflexion, and pairwise LLM-as-judge. The design below combines them.

---

## 2. The Deliberation Engine ("thinking" stage)

```
              ┌──────────────┐
 request ───▶ │ 0. Router    │── simple ──▶ fast path (1 model, 1 pass)
              └──────┬───────┘
                     │ complex / high-stakes
              ┌──────▼───────┐
              │1. Context    │  persona bible, brand voice, past posts + metrics,
              │   Builder    │  trends, continuity ledger, relevant comments (RAG)
              └──────┬───────┘
     ┌───────────────┼────────────────┐        (independent, no peeking → no anchoring)
┌────▼────┐    ┌─────▼─────┐    ┌─────▼─────┐
│Strategist│   │Trend/Hook │    │ Audience  │   + optional Contrarian
│ model A  │   │ model B   │    │ model C   │   (different families / temps)
└────┬────┘    └─────┬─────┘    └─────┬─────┘
     └──────── 2. Cross-critique (each critiques the others, structured JSON) ──┐
                                                                                  │
              ┌──────────────────────────────────────────────────────────────────▼┐
              │3. Synthesizer (strongest model): fuse the best parts, list         │
              │   which critique points it accepted or rejected and why            │
              └──────┬─────────────────────────────────────────────────────────────┘
              ┌──────▼───────┐
              │4. Verifier / │  hard checks: persona & continuity consistency,
              │   Red-team   │  platform-policy risk, claims, format specs (length,
              └──────┬───────┘  aspect, hashtags), tool-checkable facts
              ┌──────▼───────┐
              │5. Judge      │  pairwise rubric vs. alternatives → score + confidence
              └──────┬───────┘
          score ≥ τ? ├── no ──▶ loop to 2 with judge feedback (max K=2) or escalate to you
                     │ yes
              ┌──────▼───────┐
              │6. Human gate │  phone approval (tiered by risk/autonomy level)
              └──────┬───────┘
                     ▼
               execute → 7. Outcome loop (metrics at 1h/24h/7d → memory, judge calibration)
```

### Stage details

**0. Router.** A small fast model classifies the task: `trivial` (like a comment) → single pass. `standard` (caption) → 2 proposers + judge. `deep` (weekly content strategy, series concept, crisis reply) → full pipeline. This matters because you have one PC whose GPU is shared with video generation.

**1. Context Builder.** This is deterministic code, not an LLM. It pulls:
- **Persona Bible**: identity, voice, visual traits, backstory, do/don't lists. This is also the source of truth for video character consistency.
- **Continuity Ledger**: what has been shown so far (outfits, locations, storylines, running jokes). It feeds the video prompts so videos stay consistent with each other.
- Top and bottom performing past posts with their metrics, current trends, and audience comment themes (vector search).

**2. Divergent Proposers.** Each proposer gets the same context plus a *different role prompt*, and ideally runs on a *different model family*. Each one produces its answer **independently first**. Showing proposers each other's drafts too early collapses the diversity. Set temperatures to roughly 0.7–1.0.

**2b. Cross-critique.** Each model gets the *other* drafts and returns structured JSON:
`{strengths[], weaknesses[], factual_or_consistency_issues[], steal_this[], score_1_10}`.
Structured output stops critics from being vaguely polite.

**3. Synthesizer.** Your strongest model (low temperature) merges the drafts using the critiques. It must output a "decision log" of the critiques it accepted and rejected. This lets you audit the agent's thinking from your phone.

**4. Verifier.** Mostly *non-LLM* checks where possible: regex/length/spec checks, persona-ledger diff, and banned-topic list. An LLM is used only for fuzzy checks. Running an uncensored generator does **not** remove the need for this gate: the platforms aren't uncensored, and a strike or ban kills the project. The verifier encodes *platform* rules, not model-vendor morality.

**5. Judge.** Use pairwise comparison ("A or B for this audience, and why?") rather than absolute 1–10 scores, because absolute scores are noisy for LLMs. Use a different model from the synthesizer to reduce self-preference bias. Score against an explicit rubric (hook strength in the first 1.5s, persona fit, novelty vs. last 20 posts, CTA, trend fit).

**7. Outcome loop.** This is the part that compounds. Every executed decision is stored with its full deliberation trace and the real metrics later. That data is used to:
- Retrieve "what worked" as few-shot examples in future Context Builder runs.
- Check how well the judge's scores correlate with actual engagement, and adjust the rubric weights when they diverge.
- Later (phase 4): optimize prompts automatically (DSPy-style), and optionally fine-tune a small LoRA on your own accepted outputs to capture your voice.

### Deliberation config, per task type (YAML)
```yaml
caption:
  proposers: [qwen_main@0.9, mistral_alt@0.8]
  critique: false
  synthesizer: qwen_main@0.3
  judge: llama_judge
  threshold: 7.0
  max_loops: 1
weekly_strategy:
  proposers: [qwen_main@0.9, mistral_alt@0.9, llama_judge@1.0, qwen_main@1.1:contrarian]
  critique: true
  synthesizer: qwen_main@0.3
  judge: llama_judge
  threshold: 8.0
  max_loops: 2
```

---

## 3. Models (uncensored, local)

The "uncensored" variants are usually **abliterated** builds (refusal direction removed from the weights) or community fine-tunes (Dolphin, Hermes-style). Find them on Hugging Face by searching `<model> abliterated` and checking the downloads and recent comments. Quality varies: abliteration can slightly reduce reasoning ability, so test each one on your own task set (see §7).

The model landscape moves monthly, so treat these as *slots* to fill with whatever currently tops the open leaderboards:

| Slot | Role | What to look for |
|---|---|---|
| `main` | Synthesizer, strategist | Strongest reasoning model you can fit, e.g. a Qwen-family MoE (~30B total / ~3B active runs fast on consumer GPUs) or a 70B-class dense model at 4-bit if you have VRAM |
| `alt` | Proposer diversity | A **different family**: Mistral / GLM / Llama / DeepSeek-distill |
| `judge` | Judge, verifier | A third family if possible; prefer an instruction-following model over a "creative" one |
| `fast` | Router, comment triage, tagging | A 3–8B model, always loaded |
| `embed` | Memory / RAG | A small embedding model (bge / nomic / Qwen-embedding class) |
| `vlm` | Reviews generated video frames for consistency and quality | An open vision-language model (Qwen-VL class). It closes the loop with the video pipeline. |

> **Chosen setup: a single RTX 4090 (24 GB) plus cloud for some steps. See §8 for the concrete config.**

### Hardware tiers (the GPU is shared with video generation)
- **24 GB (e.g. 3090/4090):** run `fast` + one ~30B MoE at Q4. Get proposer diversity from roles and temperatures, plus one swapped-in alt model. Schedule deep thinking and video rendering at *different times*.
- **32–48 GB:** keep `main` + `alt` resident, and swap `judge` in when needed.
- **2 GPUs or 64 GB+ unified memory:** dedicate one GPU to LLMs and one to video.
- **Optional hybrid:** route non-sensitive, high-value steps (for example the weekly strategy judge) to a cloud API through the same interface. You keep everything else local.

**Inference servers:** `llama.cpp` server (GGUF, simplest, CPU offload), or `vLLM` / `TabbyAPI` (EXL quants) for throughput. Put **LiteLLM** in front of all of them so every model is an OpenAI-compatible endpoint. Swapping a model then means editing one line of config.

**GPU scheduler:** use one job queue that owns the GPU. Video renders and "deep think" jobs reserve VRAM through it, which prevents OOM crashes when both run at once.

---

## 4. System architecture

```
 Phone (PWA / Telegram)        PC browser
          │  Tailscale (private VPN, no open ports)
          ▼
 ┌─────────────────────── FastAPI backend ───────────────────────┐
 │ Orchestrator (LangGraph or a small custom state machine)       │
 │   ├─ Deliberation Engine (§2)                                  │
 │   ├─ Planner: content calendar, series arcs                    │
 │   ├─ Producer: shot list → VideoJob → video module ◀── other chat
 │   ├─ Publisher: IG Graph API / TikTok Content Posting API      │
 │   ├─ Community: comment/DM triage + reply drafts               │
 │   └─ Analyst: insights ingestion → outcome loop                │
 │ Job queue (Redis + arq/Celery) + GPU scheduler                 │
 │ Storage: Postgres + pgvector (or SQLite + sqlite-vec to start) │
 │ LiteLLM ─▶ llama.cpp / vLLM / TabbyAPI model servers           │
 └────────────────────────────────────────────────────────────────┘
```

### Interface with the video module (coordinate with the other chat)
The agent never writes raw video prompts ad hoc. It emits a structured `VideoJob`:
```json
{
  "series_id": "morning-routine",
  "character_refs": ["persona_main_v3"],
  "continuity": {"outfit": "cream knit sweater", "location": "apt_kitchen", "lighting": "golden hour"},
  "shots": [{"duration_s": 3, "camera": "handheld medium", "action": "...", "dialogue": "..."}],
  "seed_policy": "locked_per_series",
  "aspect": "9:16",
  "voice_ref": "persona_main_voice_v1"
}
```
Persona Bible + Continuity Ledger + locked character references (LoRA / reference images on the video side) is the agent's contribution to **cross-video consistency**. After rendering, the `vlm` model checks sampled frames against the reference images and sends failures back for a re-render.

### Phone control
- **PWA dashboard** over Tailscale: approval queue, deliberation traces, calendar, analytics.
- **Telegram bot** for push notifications and one-tap approve / reject / "rethink with note: …". This is the quickest path to usable phone control.

### Autonomy levels (per action type, adjustable)
- **L0**: draft only.
- **L1**: auto-execute low-risk actions (likes, simple positive-comment replies), with everything else needing approval.
- **L2**: auto-post within a pre-approved calendar.

Start everything at L0/L1 and promote action types once their approval rate is consistently above about 90%.

### Account APIs (avoid bans)
- **Instagram:** use the Graph API (requires a Professional account) for publishing Reels, comments, and insights. Avoid unofficial / private-API automation libraries, because they are the main way accounts get banned.
- **TikTok:** use the Content Posting API. Until your app passes TikTok's audit, posts are private-only, so either apply early or use "upload as draft" and publish manually from your phone.
- **AI labeling:** both platforms require realistic AI-generated content to be labeled, and both detect C2PA metadata. Build the label toggle into the Publisher. Unlabeled detection gets reach throttled or the account struck.

---

## 5. Proposed repo layout
```
agent/
  core/           orchestrator, state graph, autonomy policy
  deliberation/   router, proposers, critique, synthesizer, verifier, judge, configs/*.yaml
  memory/         persona bible, continuity ledger, vector store, outcome store
  models/         LiteLLM config, model registry, GPU scheduler client
  tools/          instagram.py, tiktok.py, trends.py, video_bridge.py
  api/            FastAPI app, auth
  ui/             PWA
  bots/           telegram
  evals/          golden tasks, judge-calibration scripts
video/            ← other chat
shared/schemas/   VideoJob, Persona, ContinuityEntry (pydantic), shared by both halves
```

---

## 6. Build phases

| Phase | Deliverable | Done when |
|---|---|---|
| **1. Foundation** (week 1–2) | LiteLLM + 2 local models, FastAPI, SQLite memory, Persona Bible editor, Telegram approve/reject | You can ask for 5 caption options from your phone and approve one |
| **2. Deliberation v1** | Router, independent proposers, synthesizer, pairwise judge, traces stored and viewable | On your eval set (§7), the pipeline beats the best single model in blind pairwise tests |
| **3. Production loop** | VideoJob schema with the video module, continuity ledger, VLM frame checks, IG publishing, TikTok drafts | One full series (≥5 videos) is produced end-to-end with consistent character and setting |
| **4. Learning loop** | Insights ingestion, outcome store, judge calibration, few-shot "what worked" retrieval, comment triage | Judge scores correlate positively with real engagement, and you tune the rubric from data |
| **5. Autonomy & polish** | Autonomy levels, PWA dashboard, GPU scheduler, prompt optimization, optional voice LoRA | Most routine actions run at L1/L2 and you mostly do approvals |

---

## 7. Measure that the "thinking" actually helps

You can't claim the swarm beats a single model without measuring it.
1. Build a **golden set** of about 30 real tasks: captions, hooks, comment replies, and series concepts.
2. For each task, produce outputs from (a) the best single local model, (b) the full pipeline, and (c) optionally a frontier private model.
3. Rate them **blind** in pairs on your phone (a simple swipe UI in the PWA).
4. Track win rate, cost (GPU-seconds) and latency per pipeline config. Cut any stage that doesn't move the win rate. This is how you find out whether critique rounds, the contrarian, or loops are worth their compute.
5. Re-run the eval every time you swap a model.

---

## 8. Concrete config: RTX 4090 (24 GB) + hybrid cloud

### Core principle: the GPU has two modes, and the cloud fills the gaps
A 14B-class video model with character references will use roughly all 24 GB, so the LLMs and video can't share the card at the same time. The GPU scheduler switches between two modes:

| | **Think mode** (GPU → LLMs) | **Render mode** (GPU → video) |
|---|---|---|
| GPU | `main` MoE (~30B total / ~3B active, Q4_K_M ≈ 17–19 GB) + embeddings (~1 GB), 16k context with q8 KV cache | Video model, ~22+ GB |
| CPU / system RAM | `fast` model (3–4B Q4) on CPU | `fast` model on CPU; `main` unloaded or running with its MoE experts offloaded to CPU (`llama.cpp --n-cpu-moe`, slow but works with ≥64 GB RAM) |
| Cloud | Proposer diversity, critique, judge | All deep thinking |
| Agent still does | Everything | Comment triage, scheduling, approvals; deep tasks go to the cloud or are queued for the next think window |

Default schedule: **render overnight in batches, think during the day.** The calendar planner produces the next day's VideoJobs in the evening so the overnight render queue is full. If your system RAM is ≥64 GB, MoE expert offloading lets `main` stay usable even in render mode.

### Local vs. cloud routing policy
Every LLM call carries a `sensitivity` tag. The router enforces it:

| Goes **local** (uncensored, private) | May go to **cloud** |
|---|---|
| Final persona-voice writing (captions, scripts, dialogue) | Trend and competitor analysis |
| Anything that would be refused by a cloud model | Weekly and monthly strategy proposers and critique |
| DMs, follower personal info, your account credentials | Judging and rubric scoring on non-sensitive drafts |
| Persona Bible edits | Eval-set judging (§7), where a strong neutral judge matters most |

**Refusal fallback:** if a cloud model refuses or waters down a response (detected with a cheap classifier on the `fast` model), the call is retried automatically on the local `main` model and the event is logged. This keeps the cloud's quality advantage without inheriting its censorship.

### Cloud provider
- **OpenRouter** (one API key, OpenAI-compatible, fits behind LiteLLM): it gives you both frontier models *and* open models from other families, including less-filtered fine-tunes. That solves the 24 GB problem of only running one local family. Cross-family proposer diversity now comes from the cloud.
- Set a **hard monthly budget cap** in LiteLLM, with per-task-type budgets. Rough expectation: a few dollars to low tens of dollars per month at the volume of a single account.
- Optional: rent a cloud GPU (RunPod / Vast) by the hour for video bursts when the overnight queue isn't enough. That decision belongs to the video side, but it uses the same GPU scheduler interface.

### Updated deliberation configs
```yaml
models:
  fast:   {where: local_cpu, model: "<3-4B instruct, Q4>"}
  main:   {where: local_gpu, model: "<~30B MoE abliterated, Q4_K_M>"}
  alt:    {where: cloud, model: "<different-family open model via OpenRouter>"}
  judge:  {where: cloud, model: "<strong frontier or large open model>"}
  vlm:    {where: local_gpu, model: "<Qwen-VL class, Q4>", mode: think}  # frame checks after render batch

caption:            # sensitive: persona voice
  proposers: [main@0.9, main@1.1:alt_role]
  synthesizer: main@0.3
  judge: judge        # judge only sees the draft, no private data
  threshold: 7.0
  max_loops: 1
comment_reply:
  route: fast         # L1 auto for simple positives; else main draft → approval
weekly_strategy:
  proposers: [main@0.9, alt@0.9, judge@1.0:contrarian]
  critique: true
  synthesizer: main@0.3   # final synthesis local so voice stays uncensored
  judge: judge
  threshold: 8.0
  max_loops: 2
```

### Revised Phase 1 checklist (4090)
1. Install `llama.cpp` server (CUDA build), then download the `fast` and `main` GGUFs and a small embedding model.
2. Set up LiteLLM with three endpoints (local-gpu, local-cpu, openrouter), a budget cap, and the refusal-fallback hook.
3. Build the GPU scheduler v0: a single Redis lock with `think` / `render` modes. Model servers are started and stopped by the scheduler.
4. Set up the FastAPI app, SQLite + sqlite-vec, and the Persona Bible.
5. Install Tailscale on the PC and phone, and set up the Telegram bot for approve / reject / rethink.
6. Milestone: send "give me 5 hooks for X" from your phone, see the deliberation trace, and approve one.
