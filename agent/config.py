"""Configuration loading: config/agent.yaml + config/tasks.yaml + environment."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent


class LlamaServerCfg(BaseModel):
    manage_processes: bool = True
    binary: str = "llama-server"
    startup_timeout_s: float = 180


class ServerCfg(BaseModel):
    model_path: str
    port: int
    host: str = "127.0.0.1"
    profiles: dict[str, list[str]] = Field(default_factory=dict)


class EndpointCfg(BaseModel):
    base_url: str
    kind: Literal["local", "cloud"]
    server: str | None = None
    uses_gpu: bool = False
    api_key_env: str | None = None
    timeout_s: float = 600
    # Qwen3 hybrid / other "thinking" models: ask the chat template to skip hidden reasoning. Otherwise a call
    # spends minutes thinking and can run out of tokens before writing the answer. Harmless for other models.
    disable_thinking: bool = True


class SlotCfg(BaseModel):
    endpoint: str
    model: str
    price_in: float = 0.0   # USD per 1M input tokens
    price_out: float = 0.0  # USD per 1M output tokens
    max_tokens: int = 2048


class RoleCfg(BaseModel):
    """A proposer lens: its instructions plus its own sampling, so one model explores differently per role."""
    prompt: str
    temperature: float | None = None   # overrides the temperature in the task's "slot@temp:role" spec
    top_p: float | None = None
    min_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None

    def sampling(self) -> dict:
        return {k: v for k, v in (("top_p", self.top_p), ("min_p", self.min_p), ("top_k", self.top_k),
                                  ("presence_penalty", self.presence_penalty)) if v is not None}


class RoutingCfg(BaseModel):
    local_fallback_slot: str = "main"
    refusal_fallback: bool = True
    confirm_refusal_with_fast: bool = True
    fallback_on_cloud_error: bool = True


class BudgetCfg(BaseModel):
    monthly_usd: float = 20.0


class ComfyCfg(BaseModel):
    url: str = "http://127.0.0.1:8188"
    free_on_think: bool = True


class InstagramCfg(BaseModel):
    enabled: bool = False
    login: Literal["facebook", "instagram"] = "facebook"
    api_version: str = "v27.0"
    user_id_env: str = "IG_USER_ID"
    token_env: str = "IG_ACCESS_TOKEN"
    public_media_base_url: str | None = None
    share_to_feed: bool = True
    poll_interval_s: float = 30
    poll_max_s: float = 900
    insights_metrics: list[str] = Field(default_factory=lambda: [
        "views", "reach", "likes", "comments", "shares", "saved", "total_interactions", "ig_reels_avg_watch_time"])


class TikTokCfg(BaseModel):
    enabled: bool = False
    mode: Literal["inbox", "direct"] = "inbox"
    client_key_env: str = "TIKTOK_CLIENT_KEY"
    client_secret_env: str = "TIKTOK_CLIENT_SECRET"
    redirect_uri: str = ""
    authorize_url: str = "https://www.tiktok.com/v2/auth/authorize/"
    scopes: list[str] = Field(default_factory=lambda: ["user.info.basic", "video.upload", "video.publish", "video.list"])
    privacy_level: str = "SELF_ONLY"
    poll_interval_s: float = 15
    poll_max_s: float = 900


class PublishingCfg(BaseModel):
    ai_label: bool = True
    caption_footer: str = ""
    posting_times: list[str] = Field(default_factory=lambda: ["12:15", "18:40"])
    instagram: InstagramCfg = Field(default_factory=InstagramCfg)
    tiktok: TikTokCfg = Field(default_factory=TikTokCfg)


class AutonomyCfg(BaseModel):
    publish: Literal["ask", "auto"] = "ask"
    comment_replies: Literal["ask", "auto_low_risk"] = "ask"


class AutopilotCfg(BaseModel):
    enabled: bool = True
    tick_s: float = 60
    comment_poll_min: float = 15
    comment_window_days: float = 14
    metric_checkpoints_h: dict[str, float] = Field(default_factory=lambda: {"1h": 1, "24h": 24, "7d": 168})


class AgentCfg(BaseModel):
    data_dir: str = "data"
    llama_server: LlamaServerCfg = Field(default_factory=LlamaServerCfg)
    servers: dict[str, ServerCfg] = Field(default_factory=dict)
    endpoints: dict[str, EndpointCfg]
    slots: dict[str, SlotCfg]
    routing: RoutingCfg = Field(default_factory=RoutingCfg)
    budget: BudgetCfg = Field(default_factory=BudgetCfg)
    # Named slot overrides; `slot_profile` picks one at load time ("local" = free, everything on the PC)
    slot_profiles: dict[str, dict[str, SlotCfg]] = Field(default_factory=dict)
    slot_profile: str | None = None
    comfyui: ComfyCfg = Field(default_factory=ComfyCfg)
    publishing: PublishingCfg = Field(default_factory=PublishingCfg)
    autonomy: AutonomyCfg = Field(default_factory=AutonomyCfg)
    autopilot: AutopilotCfg = Field(default_factory=AutopilotCfg)


class TaskCfg(BaseModel):
    description: str = ""
    sensitivity: Literal["private", "public"] = "private"
    n_options: int = 3
    proposers: list[str]
    critique: bool = False
    synthesizer: str | None = None
    judge: str | None = None
    threshold: float = 7.0
    max_loops: int = 1
    # --- variety controls (matter most when every slot is the same local model) ---
    diversify: Literal["parallel", "sequential"] = "parallel"   # sequential: later proposers avoid taken angles
    constraints: bool = False        # give each proposer a random creative constraint
    dedupe: float | None = None      # drop near-duplicate options above this cosine similarity (e.g. 0.9)
    judge_panel: int = 1             # judge as N different audience personas and average
    shuffle_for_judge: bool = True   # hide which option the synthesizer wrote / fight position bias


class TasksCfg(BaseModel):
    roles: dict[str, str | RoleCfg] = Field(default_factory=dict)
    creative_constraints: list[str] = Field(default_factory=list)
    judge_personas: list[str] = Field(default_factory=list)
    tasks: dict[str, TaskCfg]
    rubric: list[str] = Field(default_factory=list)
    default_type: str = "hooks"   # used by the router's keyword fallback


class CameraSetup(BaseModel):
    label: str
    description: str
    prompt: str
    hands_free: int = 1
    face_visible: bool = True


class Timing(BaseModel):
    min_shots: int = 2
    max_shots: int = 6
    min_shot_s: float = 2
    max_shot_s: float = 5
    min_total_s: float = 10
    max_total_s: float = 30


class FilmStyle(BaseModel):
    name: str
    timing: Timing = Field(default_factory=Timing)
    brief: str = ""
    setups: dict[str, CameraSetup]
    look: str = ""
    negative_prompt: str = ""
    banned_terms: list[str] = Field(default_factory=list)
    formats: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)


class StyleFile(BaseModel):
    active: str
    styles: dict[str, FilmStyle]

    @property
    def current(self) -> FilmStyle:
        if self.active not in self.styles:
            raise ValueError(f"config/style.yaml: active style '{self.active}' is not defined")
        return self.styles[self.active]


class Secrets(BaseModel):
    api_token: str | None = None
    telegram_token: str | None = None
    telegram_allowed_users: set[int] = Field(default_factory=set)


class Settings(BaseModel):
    root: Path
    agent: AgentCfg
    tasks: TasksCfg
    secrets: Secrets
    style_file: StyleFile | None = None

    @property
    def style(self) -> FilmStyle | None:
        return self.style_file.current if self.style_file else None

    @property
    def data_path(self) -> Path:
        p = Path(self.agent.data_dir)
        return p if p.is_absolute() else self.root / p

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def ensure_api_token(env_path: Path) -> str:
    """No password configured yet: make a strong one and save it to .env (phones pair by QR, so
    nobody has to type it)."""
    import secrets

    token = secrets.token_urlsafe(24)
    try:
        existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        lines = [ln for ln in existing.splitlines() if not ln.strip().startswith("INSTACC_API_TOKEN=")]
        lines.append(f"INSTACC_API_TOKEN={token}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass  # read-only folder: the token still works for this run
    os.environ["INSTACC_API_TOKEN"] = token
    return token


def apply_slot_profile(agent: "AgentCfg", name: str) -> None:
    """Overlay a named profile (e.g. "local" or "cloud") onto the slots."""
    if name not in agent.slot_profiles:
        raise ValueError(f"unknown slot_profile '{name}'. Known: {', '.join(agent.slot_profiles) or 'none'}")
    agent.slots = {**agent.slots, **agent.slot_profiles[name]}
    agent.slot_profile = name


def load_settings(root: Path | None = None) -> Settings:
    root = Path(root or os.environ.get("INSTACC_ROOT") or ROOT)
    load_dotenv(root / ".env")
    cfg_dir = root / "config"
    agent = AgentCfg.model_validate(yaml.safe_load((cfg_dir / "agent.yaml").read_text(encoding="utf-8")))
    tasks = TasksCfg.model_validate(yaml.safe_load((cfg_dir / "tasks.yaml").read_text(encoding="utf-8")))
    api_token = os.environ.get("INSTACC_API_TOKEN") or ensure_api_token(root / ".env")
    allowed = {
        int(x) for x in os.environ.get("TELEGRAM_ALLOWED_USERS", "").replace(" ", "").split(",") if x
    }
    if agent.slot_profile:
        apply_slot_profile(agent, agent.slot_profile)
    for name, t in tasks.tasks.items():   # fail fast on typos instead of mid-deliberation
        used = [p.split("@")[0] for p in t.proposers] + [x.split("@")[0] for x in (t.synthesizer, t.judge) if x]
        missing = sorted(set(used) - set(agent.slots))
        if missing:
            raise ValueError(f"task '{name}' uses unknown slot(s) {missing}; "
                             f"check slot_profile/slots in config/agent.yaml")
    secrets = Secrets(
        api_token=api_token,
        telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN") or None,
        telegram_allowed_users=allowed,
    )
    style_path = cfg_dir / "style.yaml"
    style_file = None
    if style_path.exists():
        style_file = StyleFile.model_validate(yaml.safe_load(style_path.read_text(encoding="utf-8")))
        style_file.current  # validate the active style exists
    return Settings(root=root, agent=agent, tasks=tasks, secrets=secrets, style_file=style_file)
