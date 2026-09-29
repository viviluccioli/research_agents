"""
Experiment 5: Enhanced Multi-Agent Debate System

This experiment adds:
1. Severity Delta (Δ) system for flaw categorization
2. Layman Translation requirement for cross-domain understanding

Does NOT include Human-in-the-Loop (see exp-5-hitl for that).
"""

from .engine import execute_debate_pipeline
from .workflow import RefereeWorkflowExp5

__all__ = ['execute_debate_pipeline', 'RefereeWorkflowExp5']
