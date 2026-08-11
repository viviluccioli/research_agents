# -*- coding: utf-8 -*-
"""
Model configuration system for multi-agent peer review.

Supports:
- Position-based model assignment (persona_1, persona_2, persona_3)
- Role-based assignment (selection, editor, debate)
- Persona-specific overrides (Econometrician, Policymaker, etc.)
- CLI override for quick testing
- YAML configuration file
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any


class ModelConfig:
    """Manages model assignments for different agents in the peer review system."""

    def __init__(self, config_path: str = "model_config.yaml"):
        """
        Initialize model configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    print(f"[ModelConfig] Loaded configuration from {self.config_path}")
                    return config
            except Exception as e:
                print(f"[ModelConfig] Warning: Failed to load {self.config_path}: {e}")
                print("[ModelConfig] Using default configuration")
                return self._default_config()
        else:
            print(f"[ModelConfig] No config file found at {self.config_path}, using defaults")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "default_model": "gpt-5.6-sol",
            "reasoning_effort": {
                "enabled": True,
                "level": "medium",
                "enable_for_rounds": [1, 2, 3]  # e.g., [1, "2a", "2b", "2c"] - excludes 0 and 3
            },
            "models": {
                "selection": {"model": None, "temperature": 0.0},
                "persona_1": {"model": None, "temperature": 0.0},
                "persona_2": {"model": None, "temperature": 0.0},
                "persona_3": {"model": None, "temperature": 0.0},
                "debate_persona_1": {"model": None, "temperature": 0.35},
                "debate_persona_2": {"model": None, "temperature": 0.35},
                "debate_persona_3": {"model": None, "temperature": 0.35},
                "editor": {"model": None, "temperature": 0.0}
            },
            "persona_overrides": {}
        }

    def get_model(
        self,
        role: str,
        position: Optional[int] = None,
        persona_name: Optional[str] = None,
        default_override: Optional[str] = None
    ) -> str:
        """
        Get model for a specific role.

        Priority order:
        1. Persona-specific override (if persona_name provided)
        2. Position-based config (e.g., persona_1, debate_persona_2)
        3. Role-based config (e.g., selection, editor)
        4. CLI override (default_override) - only replaces the default model
        5. Default model from config

        Args:
            role: "selection", "persona", "debate", or "editor"
            position: 0, 1, 2 (for persona_1, persona_2, persona_3)
            persona_name: "Econometrician", "Policymaker", etc.
            default_override: CLI-provided model to use instead of the configured default

        Returns:
            Model identifier string
        """
        # Priority 1: Persona-specific override
        if persona_name:
            persona_overrides = self.config.get("persona_overrides", {})
            if persona_name in persona_overrides:
                override_model = persona_overrides[persona_name].get("model")
                if override_model:
                    return override_model

        # Priority 2: Position-based config
        if position is not None:
            key = f"{role}_persona_{position + 1}"
            models = self.config.get("models", {})
            if key in models:
                model = models[key].get("model")
                if model:
                    return model

        # Priority 3: Role-based config
        models = self.config.get("models", {})
        if role in models:
            model = models[role].get("model")
            if model:
                return model

        # Priority 4: CLI override (only replaces default model, not specific overrides)
        if default_override:
            return default_override

        # Priority 5: Default
        return self.config.get("default_model", "gpt-5.6-sol")

    def get_temperature(
        self,
        role: str,
        position: Optional[int] = None,
        persona_name: Optional[str] = None
    ) -> float:
        """
        Get temperature for a specific role.

        Args:
            role: "selection", "persona", "debate", or "editor"
            position: 0, 1, 2 (for persona_1, persona_2, persona_3)
            persona_name: Optional persona name for overrides

        Returns:
            Temperature value (float)
        """
        # Check persona-specific override first
        if persona_name:
            persona_overrides = self.config.get("persona_overrides", {})
            if persona_name in persona_overrides:
                temp = persona_overrides[persona_name].get("temperature")
                if temp is not None:
                    return float(temp)

        # Check position-based config
        if position is not None:
            key = f"{role}_persona_{position + 1}"
            models = self.config.get("models", {})
            if key in models:
                temp = models[key].get("temperature")
                if temp is not None:
                    return float(temp)

        # Check role-based config
        models = self.config.get("models", {})
        if role in models:
            temp = models[role].get("temperature")
            if temp is not None:
                return float(temp)

        # Default temperatures by role
        defaults = {
            "selection": 0.0,
            "persona": 0.0,
            "debate": 0.35,
            "editor": 0.0
        }
        return defaults.get(role, 0.0)

    def get_reasoning_effort_config(self) -> Dict[str, Any]:
        """
        Get reasoning effort (extended thinking) configuration.

        Returns:
            Dictionary with 'enabled' (bool), 'level' (str), and 'enable_for_rounds' (list) keys
        """
        return self.config.get("reasoning_effort", {
            "enabled": False,
            "level": "none",
            "enable_for_rounds": []
        })

    def should_use_reasoning_effort(self, round_id: str) -> str:
        """
        Check if reasoning effort should be enabled for a specific round.

        Args:
            round_id: "0", "1", "2a", "2b", "2c", or "3"

        Returns:
            Reasoning effort level if enabled for the round, otherwise "none".
        """
        config = self.get_reasoning_effort_config()
        if not config.get("enabled"):
            return "none"

        enable_for_rounds = config.get("enable_for_rounds", [])
        # Convert round_id to match config format (string or int)
        if str(round_id) in [str(r) for r in enable_for_rounds]:
            return config.get("level", "high")
        return "none"

    def print_config_summary(self):
        """Print a summary of the current configuration."""
        print("\n" + "="*80)
        print("MODEL CONFIGURATION SUMMARY")
        print("="*80)
        print(f"Default Model: {self.config.get('default_model')}")

        # Reasoning effort configuration
        reasoning_config = self.get_reasoning_effort_config()
        print(f"\nReasoning Effort (Extended Thinking):")
        print(f"  Enabled: {reasoning_config.get('enabled')}")
        print(f"  Level: {reasoning_config.get('level')}")
        print(f"  Enable for rounds: {reasoning_config.get('enable_for_rounds')}")
        if reasoning_config.get('enabled'):
            print(f"  Note: Requires temperature=1.0 (automatically enforced)")

        print("\nRole-Based Models:")

        models = self.config.get("models", {})
        for role in ["selection", "persona_1", "persona_2", "persona_3",
                     "debate_persona_1", "debate_persona_2", "debate_persona_3", "editor"]:
            if role in models:
                model = models[role].get("model", "default")
                temp = models[role].get("temperature", "default")
                print(f"  {role:20s}: {model} (temp: {temp})")

        persona_overrides = self.config.get("persona_overrides", {})
        if persona_overrides:
            print("\nPersona-Specific Overrides:")
            for persona, settings in persona_overrides.items():
                model = settings.get("model", "default")
                temp = settings.get("temperature", "default")
                print(f"  {persona:20s}: {model} (temp: {temp})")

        print("="*80 + "\n")


# Test/demo usage
if __name__ == "__main__":
    print("Testing ModelConfig class...\n")

    # Test 1: Default config (no file)
    print("Test 1: Default configuration")
    config = ModelConfig("nonexistent.yaml")
    config.print_config_summary()

    # Test 2: Get models with different priorities
    print("\nTest 2: Model resolution priority")

    # No overrides
    model = config.get_model("persona", position=0)
    print(f"persona position=0, no overrides: {model}")

    # With CLI override
    model = config.get_model("persona", position=0, default_override="opus-4-8")
    print(f"persona position=0, CLI override='opus-4-8': {model}")

    # Test 3: Temperature resolution
    print("\nTest 3: Temperature resolution")
    temp = config.get_temperature("persona", position=0)
    print(f"persona position=0 temperature: {temp}")

    temp = config.get_temperature("debate", position=1)
    print(f"debate position=1 temperature: {temp}")

    temp = config.get_temperature("editor")
    print(f"editor temperature: {temp}")

    print("\n✓ ModelConfig tests complete!")
