"""LLM doctor: why is output slow or nonsensical? Run while the agent is running (llm-doctor.bat).

Checks model files, GPU offload (from llama-server logs), chat template, speed, and whether each model
writes sensible text with plain settings vs. the configured role settings. Writes data/logs/doctor.txt.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import RoleCfg, load_settings  # noqa: E402
from agent.deliberation.quality import garbled_reason  # noqa: E402

lines: list[str] = []
problems: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def problem(s: str) -> None:
    problems.append(s)
    out("  !! " + s)


def log_facts(log: Path) -> dict:
    if not log.exists():
        return {}
    txt = log.read_text(encoding="utf-8", errors="replace")[-400_000:]
    f: dict = {}
    if m := re.findall(r"found (\d+) CUDA devices", txt):
        f["cuda_devices"] = int(m[-1])
    if m := re.findall(r"offloaded (\d+)/(\d+) layers to GPU", txt):
        f["offloaded"] = tuple(map(int, m[-1]))
    if m := re.findall(r"(CUDA0|CPU_Mapped|CPU) model buffer size\s*=\s*([\d.]+) MiB", txt):
        f["buffers"] = {k: float(v) for k, v in m[-4:]}
    if m := re.findall(r"n_ctx\s*=\s*(\d+)", txt):
        f["n_ctx"] = int(m[-1])
    if "out of memory" in txt.lower():
        f["oom"] = True
    return f


def chat(client: httpx.Client, base: str, messages, **params) -> tuple[str, float | None, float]:
    t0 = time.time()
    r = client.post(base + "/chat/completions", json={"model": "x", "messages": messages, **params}, timeout=300)
    r.raise_for_status()
    d = r.json()
    text = (d["choices"][0]["message"].get("content") or "").strip()
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)
    tps = (d.get("timings") or {}).get("predicted_per_second")
    return text, tps, time.time() - t0


def main() -> int:
    s = load_settings()
    a = s.agent
    out(f"instacc LLM doctor  ·  {time.strftime('%Y-%m-%d %H:%M')}  ·  slot_profile={a.slot_profile}")
    out("=" * 72)

    out("\n[1] Model files")
    for name, srv in a.servers.items():
        p = s.resolve(srv.model_path)
        if p.exists():
            gb = p.stat().st_size / 1e9
            out(f"  {name:5} {p.name}  {gb:.1f} GB")
            if name == "main" and gb < 10:
                problem(f"main model is only {gb:.1f} GB; expected ~18 GB (Qwen3-30B-A3B Q4_K_M). Wrong or incomplete file?")
        else:
            problem(f"{name}: model file missing at {p} (run get-llm-models.bat)")

    out("\n[2] Config sampling")
    for rname, r in s.tasks.roles.items():
        rc = r if isinstance(r, RoleCfg) else RoleCfg(prompt=r)
        flags = []
        if rc.temperature and rc.temperature > 1.0:
            flags.append(f"temperature {rc.temperature} > 1.0")
        if rc.presence_penalty:
            flags.append(f"presence_penalty {rc.presence_penalty} (breaks JSON on llama.cpp)")
        out(f"  {rname:11} temp={rc.temperature} " + ("  <-- " + "; ".join(flags) if flags else "ok"))
        for f in flags:
            problem(f"role '{rname}': {f}")

    client = httpx.Client()
    for name in ("main", "fast"):
        srv = a.servers.get(name)
        if not srv:
            continue
        base = f"http://{srv.host}:{srv.port}"
        out(f"\n[3] Server '{name}' at {base}")
        try:
            h = client.get(base + "/health", timeout=5)
            if h.status_code != 200:
                problem(f"{name}: /health returned {h.status_code} (still loading? GPU in render mode?)")
                continue
        except httpx.HTTPError:
            problem(f"{name}: not reachable. Is start-agent.bat running, and is the GPU in Thinking mode?")
            continue
        try:
            props = client.get(base + "/props", timeout=10).json()
            tmpl = props.get("chat_template") or ""
            out(f"  model: {Path(str(props.get('model_path', '?'))).name}")
            out(f"  context: {(props.get('default_generation_settings') or {}).get('n_ctx', '?')}")
            out(f"  chat template: {'present' if tmpl else 'MISSING'}" + (" (Qwen/ChatML)" if "<|im_start|>" in tmpl else ""))
            if not tmpl:
                problem(f"{name}: no chat template; start llama-server with --jinja and a GGUF that embeds its template")
        except Exception as e:
            out(f"  (props unavailable: {e})")
        lf = log_facts(s.data_path / "logs" / f"{name}.log")
        if lf:
            out(f"  log: {lf}")
            if name == "main":
                if lf.get("cuda_devices") == 0:
                    problem("main: llama.cpp found NO CUDA device, so it runs on the CPU (very slow). Use the CUDA build + cudart DLLs.")
                off = lf.get("offloaded")
                if off and off[0] < off[1]:
                    problem(f"main: only {off[0]}/{off[1]} layers on the GPU; the rest run on the CPU (slow). "
                            "Lower -c in config/agent.yaml or close other GPU apps (ComfyUI).")
                if lf.get("oom"):
                    problem("main: the log mentions out-of-memory")

        msgs = [{"role": "system", "content": "You are a helpful assistant. Answer in English."},
                {"role": "user", "content": "Write two short, funny sentences about failing at latte art."}]
        no_think = {"chat_template_kwargs": {"enable_thinking": False}}
        try:   # does this model "think" by default? (the agent switches it off per request)
            r = client.post(base + "/v1/chat/completions", json={"model": "x", "messages": msgs, "max_tokens": 60,
                                                                  "temperature": 0.7}, timeout=300).json()
            m = r["choices"][0]["message"]
            if m.get("reasoning_content") or "<think>" in (m.get("content") or ""):
                out("  thinking model: YES. Hidden reasoning is on by default; the agent turns it off per request "
                    "(disable_thinking in config/agent.yaml).")
                if name == "main" and "2507" not in str(s.resolve(srv.model_path).name):
                    out("     note: the planned main model is Qwen3-30B-A3B-Instruct-2507 (non-thinking).")
            else:
                out("  thinking model: no")
        except Exception as e:
            out(f"  (thinking probe failed: {e})")
        try:
            text, tps, secs = chat(client, base + "/v1", msgs, temperature=0.7, max_tokens=160, **no_think)
            why = garbled_reason(text)
            out(f"  plain test ({secs:.1f}s, {tps or 0:.1f} tok/s): {text[:200]!r}")
            if why:
                problem(f"{name}: garbled even with plain settings ({why}). Suspect the model file or chat template; "
                        "re-download the GGUF or try another quant.")
            if name == "main" and tps and tps < 20:
                problem(f"main: {tps:.1f} tok/s is slow for a 3090 (expect ~60-120). Check GPU offload above.")
            if name == "fast" and tps and tps < 8:
                problem(f"fast: {tps:.1f} tok/s on CPU; consider a smaller fast model or fewer CPU threads in use")
        except Exception as e:
            problem(f"{name}: plain test failed: {e}")
            continue

        role = s.tasks.roles.get("hook" if name == "main" else "wildcard")
        if role is not None:
            rc = role if isinstance(role, RoleCfg) else RoleCfg(prompt=role)
            jmsgs = [{"role": "system", "content": "Reply with JSON only. Write in English."},
                     {"role": "user", "content": 'Give 3 short hooks for a video about failing at latte art. '
                                                 'JSON: {"options": [{"text": "..."}]}'}]
            try:
                text, tps, secs = chat(client, base + "/v1", jmsgs, temperature=rc.temperature or 0.8, max_tokens=300,
                                       response_format={"type": "json_object"}, **no_think, **rc.sampling())
                import json as _json
                try:
                    opts = [o.get("text", "") for o in _json.loads(text).get("options", [])]
                except Exception:
                    opts = []
                    problem(f"{name}: JSON test with role settings did not return valid JSON: {text[:150]!r}")
                bad = [o for o in opts if garbled_reason(o)]
                out(f"  role-settings JSON test ({secs:.1f}s): {opts}")
                if bad:
                    problem(f"{name}: garbled with the configured role settings but not with plain ones. "
                            "Lower temperatures in config/tasks.yaml")
            except Exception as e:
                problem(f"{name}: JSON test failed: {e}")

    out("\n" + "=" * 72)
    if problems:
        out(f"{len(problems)} problem(s) found:")
        for p in problems:
            out(" - " + p)
    else:
        out("No problems found: models load on the GPU, write sensible text, and the config sampling is safe.")
    rep = s.data_path / "logs" / "doctor.txt"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out(f"\nReport saved to {rep}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
