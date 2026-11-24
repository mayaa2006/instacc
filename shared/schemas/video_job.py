"""VideoJob: the agent -> video pipeline contract. DRAFT v0.2, to be agreed with the video side.

v0.2 (phone-realism): each shot has `setup` (how the one phone is held: selfie / propped / pov / mirror /
screen), `camera_motion`, and a ready-to-use `render_prompt`; the job has `style` and `negative_prompt`.
The video side should feed `render_prompt` + `negative_prompt` to the model as-is.

The agent fills in the creative fields (series, characters, continuity, shots). The video side
owns `render` (workflow, resolution, steps...) and reports back via `POST /api/posts/{id}/rendered`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "0.2"


class Continuity(BaseModel):
    """State that must match previous videos in the series."""
    model_config = ConfigDict(extra="allow")
    outfit: str = ""
    location: str = ""
    lighting: str = ""
    props: list[str] = Field(default_factory=list)
    time_of_day: str = ""
    notes: str = ""


class Shot(BaseModel):
    model_config = ConfigDict(extra="allow")
    duration_s: float = Field(ge=0.5, le=15)
    camera: str = Field(description="framing + movement, e.g. 'handheld medium close-up, slow push in'")
    action: str = Field(description="what happens, concrete and visual")
    setting: str = ""
    dialogue: str = Field("", description="spoken line(s), empty if none")
    on_screen_text: str = ""
    keyframe_prompt: str = Field("", description="still-image prompt for the first frame (I2V keyframe)")
    continue_from_previous: bool = Field(True, description="seed this shot from the previous shot's last frame")
    setup: str = Field("", description="how the single phone is held; an id from config/style.yaml setups")
    camera_motion: str = Field("", description="e.g. 'slight handheld shake', 'static', 'she tilts the phone down'")
    render_prompt: str = Field("", description="final text prompt for the video model (built by the agent)")


class Audio(BaseModel):
    model_config = ConfigDict(extra="allow")
    voice_ref: str | None = None
    music_mood: str = ""
    sfx: list[str] = Field(default_factory=list)


class VideoJob(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: str = SCHEMA_VERSION
    post_id: str | None = None
    series_id: str | None = None
    title: str = ""
    concept: str = ""
    character_refs: list[str] = Field(default_factory=list, description="Persona.visual.reference_ids")
    continuity: Continuity = Field(default_factory=Continuity)
    shots: list[Shot] = Field(min_length=1, max_length=12)
    aspect: Literal["9:16", "1:1", "4:5", "16:9"] = "9:16"
    seed_policy: Literal["locked_per_series", "locked_per_video", "random"] = "locked_per_series"
    audio: Audio = Field(default_factory=Audio)
    style: str = Field("", description="filming style id from config/style.yaml, e.g. phone_ugc")
    negative_prompt: str = Field("", description="what the video model must avoid (built from the style)")
    realism_warnings: list[str] = Field(default_factory=list, description="issues the agent couldn't fix")
    render: dict = Field(default_factory=dict, description="owned by the video side")

    @field_validator("shots")
    @classmethod
    def _first_shot_has_keyframe(cls, shots: list[Shot]) -> list[Shot]:
        if shots and not shots[0].keyframe_prompt:
            shots[0].keyframe_prompt = shots[0].action
        return shots

    @property
    def total_duration_s(self) -> float:
        return round(sum(s.duration_s for s in self.shots), 2)
