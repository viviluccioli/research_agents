"""
Prompt Loader for Multi-Agent Debate System

This module loads versioned prompt files based on the configuration in config.yaml.
It provides a clean interface for accessing prompts without hardcoding them in the code.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class PromptLoader:
    """Loads and manages versioned prompts for the multi-agent debate system."""

    def __init__(self, config_path: str = None):
        """
        Initialize the prompt loader.

        Args:
            config_path: Path to config.yaml. If None, uses default location.
        """
        if config_path is None:
            # Default to config.yaml in the same directory as this file
            self.base_dir = Path(__file__).parent
            config_path = self.base_dir / "config.yaml"
        else:
            config_path = Path(config_path)
            self.base_dir = config_path.parent

        self.config = self._load_config(config_path)
        self._prompt_cache = {}

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_prompt_file(self, file_path: Path) -> str:
        """Load a prompt file and return its contents."""
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_error_severity_prompt(self) -> str:
        """
        Get the shared error severity classification block.

        Returns:
            The error severity prompt as a string
        """
        cache_key = "additional_context_error_severity"
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        config = self.config['additional_context']['error_severity']
        version = config['version']
        file_path = self.base_dir / config['file'].format(version=version)
        prompt = self._load_prompt_file(file_path)

        self._prompt_cache[cache_key] = prompt
        return prompt

    def get_paper_type_context(self, paper_type: str) -> str:
        """
        Get the persona selection guidance for a given paper type.

        Args:
            paper_type: One of "empirical", "theoretical", "policy"

        Returns:
            The paper type context prompt as a string, or "" if not found
        """
        if not paper_type:
            return ""

        cache_key = f"additional_context_paper_type_{paper_type}"
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        paper_type_configs = self.config['additional_context'].get('paper_type_contexts', {})
        if paper_type not in paper_type_configs:
            return ""

        config = paper_type_configs[paper_type]
        version = config['version']
        file_path = self.base_dir / config['file'].format(version=version)

        try:
            prompt = self._load_prompt_file(file_path)
        except FileNotFoundError:
            return ""

        self._prompt_cache[cache_key] = prompt
        return prompt

    def get_persona_prompt(self, persona_name: str) -> str:
        """
        Get the system prompt for a specific persona, with error severity injected.

        Args:
            persona_name: Name of the persona (e.g., "theorist", "empiricist")

        Returns:
            The persona's system prompt as a string
        """
        cache_key = f"persona_{persona_name}"
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        persona_name_lower = persona_name.lower()
        if persona_name_lower not in self.config['personas']:
            raise ValueError(f"Unknown persona: {persona_name}")

        persona_config = self.config['personas'][persona_name_lower]
        version = persona_config['version']
        file_template = persona_config['file']

        # Substitute version into file path
        file_path = self.base_dir / file_template.format(version=version)
        prompt = self._load_prompt_file(file_path)

        # Inject error severity block in place of {error_severity} placeholder
        prompt = prompt.replace("{error_severity}", self.get_error_severity_prompt())

        # Cache for future use
        self._prompt_cache[cache_key] = prompt
        return prompt

    def get_debate_round_prompt(self, round_name: str) -> str:
        """
        Get the prompt for a specific debate round.

        Args:
            round_name: Name of the round (e.g., "round_0_selection", "round_2a_cross_exam")

        Returns:
            The debate round prompt as a string
        """
        cache_key = f"round_{round_name}"
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        if round_name not in self.config['debate_rounds']:
            raise ValueError(f"Unknown debate round: {round_name}")

        round_config = self.config['debate_rounds'][round_name]
        version = round_config['version']
        file_template = round_config['file']

        # Substitute version into file path
        file_path = self.base_dir / file_template.format(version=version)
        prompt = self._load_prompt_file(file_path)

        # Cache for future use
        self._prompt_cache[cache_key] = prompt
        return prompt

    def get_all_persona_prompts(self) -> Dict[str, str]:
        """
        Get all persona system prompts.

        Returns:
            Dictionary mapping persona names (capitalized) to their prompts
        """
        prompts = {}
        for persona_name in self.config['personas'].keys():
            # Capitalize first letter to match original format
            capitalized_name = persona_name.capitalize()
            prompts[capitalized_name] = self.get_persona_prompt(persona_name)
        return prompts

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration settings.

        Returns:
            Dictionary with model configuration
        """
        return self.config.get('model', {})

    def reload_prompts(self):
        """Clear cache and reload all prompts. Useful for testing prompt changes."""
        self._prompt_cache.clear()
        self.config = self._load_config(self.base_dir / "config.yaml")


# Singleton instance for easy importing
_loader_instance = None


def get_prompt_loader() -> PromptLoader:
    """Get the singleton prompt loader instance."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = PromptLoader()
    return _loader_instance
