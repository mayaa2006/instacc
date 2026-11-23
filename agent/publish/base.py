from __future__ import annotations

from dataclasses import dataclass, field


class PublishError(RuntimeError):
    pass


class AuthError(PublishError):
    """Token missing/expired: the user must reconnect the account."""


@dataclass
class PublishResult:
    platform: str
    external_id: str | None          # id used for metrics/comments (IG media id / TikTok public post id)
    url: str | None = None
    state: str = "published"         # published | in_drafts | private | processing
    extra: dict = field(default_factory=dict)
