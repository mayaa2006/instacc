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


class SlotCfg(BaseModel):
    endpoint: str
    model: str
    price_in: float = 0.0   # USD per 1M input tokens
    price_out: float = 0.0  # USD per 1M output tokens
    max_tokens: int = 2048


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


class AgentCfg(BaseModel):
    data_dir: str = "data"
    llama_server: LlamaServerCfg = Field(default_factory=LlamaServerCfg)
    servers: dict[str, ServerCfg] = Field(default_factory=dict)
    endpoints: dict[str, EndpointCfg]
    slots: dict[str, SlotCfg]
    routing: RoutingCfg = Field(default_factory=RoutingCfg)
    budget: BudgetCfg = Field(default_factory=BudgetCfg)
    comfyui: ComfyCfg = Field(default_factory=ComfyCfg)


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


class TasksCfg(BaseModel):
    roles: dict[str, str] = Field(default_factory=dict)
    tasks: dict[str, TaskCfg]
    rubric: list[str] = Field(default_factory=list)
    default_type: str = "hooks"   # used by the router's keyword fallback


class Secrets(BaseModel):
    api_token: str | None = None
    telegram_token: str | None = None
    telegram_allowed_users: set[int] = Field(default_factory=set)


class Settings(BaseModel):
    root: Path
    agent: AgentCfg
    tasks: TasksCfg
    secrets: Secrets

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


def load_settings(root: Path | None = None) -> Settings:
    root = Path(root or os.environ.get("INSTACC_ROOT") or ROOT)
    load_dotenv(root / ".env")
    cfg_dir = root / "config"
    agent = AgentCfg.model_validate(yaml.safe_load((cfg_dir / "agent.yaml").read_text(encoding="utf-8")))
    tasks = TasksCfg.model_validate(yaml.safe_load((cfg_dir / "tasks.yaml").read_text(encoding="utf-8")))
    allowed = {
        int(x) for x in os.environ.get("TELEGRAM_ALLOWED_USERS", "").replace(" ", "").split(",") if x
    }
    secrets = Secrets(
        api_token=os.environ.get("INSTACC_API_TOKEN") or None,
        telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN") or None,
        telegram_allowed_users=allowed,
    )
    return Settings(root=root, agent=agent, tasks=tasks, secrets=secrets)
