"""
Prompt Loader for Section Evaluator

This module loads versioned prompt files for the section evaluator based on config.yaml.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class SectionEvaluatorPromptLoader:
    """Loads and manages versioned prompts for the section evaluator."""

    def __init__(self, config_path: str = None):
        """
        Initialize the prompt loader.

        Args:
            config_path: Path to config.yaml. If None, uses default location.
        """
        if config_path is None:
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

    def get_paper_type_context(self, paper_type: str) -> str:
        """
        Get the context prompt for a specific paper type.

        Args:
            paper_type: Name of the paper type (e.g., "empirical", "theoretical")

        Returns:
            The paper type context prompt as a string
        """
        cache_key = f"paper_type_{paper_type}"
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        if paper_type not in self.config['paper_type_contexts']:
            # Return empty string if paper type not found (graceful degradation)
            return ""

        config_entry = self.config['paper_type_contexts'][paper_type]
        version = config_entry['version']
        file_template = config_entry['file']

        file_path = self.base_dir / file_template.format(version=version)
        prompt = self._load_prompt_file(file_path)

        self._prompt_cache[cache_key] = prompt
        return prompt

    def get_section_type_guidance(self, section_type: str) -> str:
        """
        Get the guidance prompt for a specific section type.

        Args:
            section_type: Name of the section type (e.g., "introduction", "methodology")

        Returns:
            The section type guidance prompt as a string
        """
        cache_key = f"section_type_{section_type}"
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        if section_type not in self.config['section_type_guidance']:
            # Return empty string if section type not found (graceful degradation)
            return ""

        config_entry = self.config['section_type_guidance'][section_type]
        version = config_entry['version']
        file_template = config_entry['file']

        file_path = self.base_dir / file_template.format(version=version)
        prompt = self._load_prompt_file(file_path)

        self._prompt_cache[cache_key] = prompt
        return prompt

    def get_master_prompt(self, prompt_name: str) -> str:
        """
        Get a master prompt template.

        Args:
            prompt_name: Name of the master prompt (e.g., "scoring_philosophy", "task_instructions")

        Returns:
            The master prompt as a string
        """
        cache_key = f"master_{prompt_name}"
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        if prompt_name not in self.config['master_prompts']:
            raise ValueError(f"Unknown master prompt: {prompt_name}")

        config_entry = self.config['master_prompts'][prompt_name]
        version = config_entry['version']
        file_template = config_entry['file']

        file_path = self.base_dir / file_template.format(version=version)
        prompt = self._load_prompt_file(file_path)

        self._prompt_cache[cache_key] = prompt
        return prompt

    def get_all_paper_type_contexts(self) -> Dict[str, str]:
        """
        Get all paper type context prompts.

        Returns:
            Dictionary mapping paper type names to their context prompts
        """
        contexts = {}
        for paper_type in self.config['paper_type_contexts'].keys():
            contexts[paper_type] = self.get_paper_type_context(paper_type)
        return contexts

    def get_all_section_type_guidance(self) -> Dict[str, str]:
        """
        Get all section type guidance prompts.

        Returns:
            Dictionary mapping section type names to their guidance prompts
        """
        guidance = {}
        for section_type in self.config['section_type_guidance'].keys():
            guidance[section_type] = self.get_section_type_guidance(section_type)
        return guidance

    def reload_prompts(self):
        """Clear cache and reload all prompts. Useful for testing prompt changes."""
        self._prompt_cache.clear()
        self.config = self._load_config(self.base_dir / "config.yaml")


# Singleton instance for easy importing
_loader_instance = None


def get_section_evaluator_prompt_loader() -> SectionEvaluatorPromptLoader:
    """Get the singleton prompt loader instance."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = SectionEvaluatorPromptLoader()
    return _loader_instance
