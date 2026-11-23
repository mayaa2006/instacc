"""Persona Bible: single source of truth for the character.

Shared by the agent (voice, content rules) and the video module (visual
traits, reference IDs) so both halves stay consistent.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Identity(BaseModel):
    model_config = ConfigDict(extra="allow")
    age: int | None = None
    location: str = ""
    background: str = ""


class Voice(BaseModel):
    model_config = ConfigDict(extra="allow")
    tone: str = ""
    vocabulary: str = ""
    emoji_use: str = ""
    sentence_style: str = ""


class Visual(BaseModel):
    model_config = ConfigDict(extra="allow")
    appearance: str = ""
    signature_style: str = ""
    reference_ids: list[str] = Field(default_factory=list)


class Persona(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str
    handle: str = ""
    tagline: str = ""
    identity: Identity = Field(default_factory=Identity)
    voice: Voice = Field(default_factory=Voice)
    visual: Visual = Field(default_factory=Visual)
    content_pillars: list[str] = Field(default_factory=list)
    audience: str = ""
    do: list[str] = Field(default_factory=list)
    dont: list[str] = Field(default_factory=list)
    example_posts: list[str] = Field(default_factory=list)

    def to_prompt(self) -> str:
        """Compact text block for system prompts."""
        lines = [f"PERSONA: {self.name} {self.handle}".rstrip()]
        if self.tagline:
            lines.append(f"Tagline: {self.tagline}")
        ident = self.identity
        bits = [str(x) for x in (f"{ident.age}" if ident.age else "", ident.location) if x]
        if bits or ident.background:
            lines.append(f"Identity: {', '.join(bits)}. {ident.background}".strip())
        v = self.voice
        lines.append(
            f"Voice: tone={v.tone}; vocabulary={v.vocabulary}; emoji={v.emoji_use}; style={v.sentence_style}"
        )
        if self.visual.appearance or self.visual.signature_style:
            lines.append(f"Look: {self.visual.appearance}; style: {self.visual.signature_style}")
        if self.content_pillars:
            lines.append("Content pillars: " + ", ".join(self.content_pillars))
        if self.audience:
            lines.append(f"Audience: {self.audience}")
        if self.do:
            lines.append("DO: " + "; ".join(self.do))
        if self.dont:
            lines.append("DON'T: " + "; ".join(self.dont))
        if self.example_posts:
            lines.append("Example posts:\n" + "\n".join(f"- {p}" for p in self.example_posts[:5]))
        return "\n".join(lines)
