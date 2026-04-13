"""
Referee Classifier Package

Experimental referee report pipeline with automatic paper classification
and adaptive persona selection.
"""

from referee._utils.paper_classifier import classify_paper, PaperClassification
from referee.personas import adjust_persona_weights
from referee.engine import execute_debate_pipeline

__all__ = [
    'classify_paper',
    'PaperClassification',
    'adjust_persona_weights',
    'execute_debate_pipeline'
]

__version__ = '0.1.0'
