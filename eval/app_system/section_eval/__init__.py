"""
section_eval — modular paper-type-aware section evaluation package.
"""

from .main import SectionEvaluatorApp
from .evaluator import SectionEvaluator
from .criteria.base import PAPER_TYPES, PAPER_TYPE_LABELS, SECTION_DEFAULTS

__all__ = [
    "SectionEvaluatorApp",
    "SectionEvaluator",
    "PAPER_TYPES",
    "PAPER_TYPE_LABELS",
    "SECTION_DEFAULTS",
]
