"""
Memo Evaluation System

A parallel system for evaluating policy memos using Multi-Agent Debate architecture
with memo-specific analyst personas.

Main components:
- memo_engine: Debate orchestration for memo evaluation
- memo_prompts: Memo-specific analyst personas
"""

from .memo_engine import execute_debate_pipeline
from .memo_prompts import MEMO_SYSTEM_PROMPTS

__all__ = [
    'execute_debate_pipeline',
    'MEMO_SYSTEM_PROMPTS',
]
