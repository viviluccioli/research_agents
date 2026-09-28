"""Single configuration source. Runtime credentials come only from the environment."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from schemas import Schema, Text
from token_tracker import Pricing


class StageSettings(Schema):
    model: Optional[Text] = None
    temperature: float = Field(ge=0, le=2, default=0.0)


class Stages(Schema):
    selection: StageSettings = Field(default_factory=StageSettings)
    pre: StageSettings = Field(default_factory=StageSettings)
    debate: StageSettings = Field(default_factory=lambda: StageSettings(temperature=0.35))
    post: StageSettings = Field(default_factory=StageSettings)
    editor: StageSettings = Field(default_factory=StageSettings)


class Settings(Schema):
    # Inherited benchmark identifier; choose a model available on your endpoint.
    default_model: Text = "gpt-5.6-sol"
    reasoning_effort: Literal["none"] = "none"
    stages: Stages = Field(default_factory=Stages)
    debate_rounds: int = Field(ge=0, default=2)
    max_attempts: int = Field(ge=1, default=3)
    max_output_tokens: int = Field(ge=1, default=12000)
    timeout_seconds: float = Field(gt=0, default=120.0)
    max_context_characters: int = Field(ge=1, default=300000)
    manuscript_characters: int = Field(ge=1, default=180000)
    selection_characters: int = Field(ge=1, default=30000)
    editor_characters: int = Field(ge=1, default=40000)
    pricing: dict[str, Pricing] = Field(default_factory=dict)


class ModelConfig:
    def __init__(self, config_path=None, *, overrides=None):
        if config_path is not None and overrides is not None:
            raise ValueError("Use either a settings path or overrides")
        raw = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path is not None else dict(overrides or {})
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a JSON object")
        if "PEER_REVIEW_MODEL" in os.environ:
            raw["default_model"] = os.environ["PEER_REVIEW_MODEL"]
        self.settings = Settings.model_validate(raw)

    def _stage(self, stage):
        name = stage.lower()
        if name not in Stages.model_fields:
            raise ValueError(f"Unknown review stage: {stage}")
        return getattr(self.settings.stages, name)

    def get_model(self, stage, default_override=None):
        return self._stage(stage).model or default_override or self.settings.default_model

    def get_temperature(self, stage):
        return self._stage(stage).temperature

    def snapshot(self):
        return self.settings.model_dump(mode="json")
