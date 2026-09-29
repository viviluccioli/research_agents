"""
Referee Report Package

This package contains the multi-agent debate (MAD) system for generating referee reports.

Main Production Components:
- workflow: Main referee report UI (RefereeWorkflow)
- engine: Multi-agent debate orchestration (execute_debate_pipeline)

Internal Components:
- _utils: Helper functions and utilities
- _archived: Alternate implementations (not main code path)
"""

# Import main production classes
from referee.workflow import RefereeWorkflow
from referee.engine import (
    execute_debate_pipeline,
    SELECTION_PROMPT,
    SYSTEM_PROMPTS,
    DEBATE_PROMPTS
)

# Backward compatibility aliases (deprecated - use RefereeWorkflow instead)
RefereeReportCheckerSummarized = RefereeWorkflow
RefereeReportChecker = RefereeWorkflow  # Main workflow is now the default

__all__ = [
    # Main production API
    'RefereeWorkflow',
    'execute_debate_pipeline',
    'SELECTION_PROMPT',
    'SYSTEM_PROMPTS',
    'DEBATE_PROMPTS',
    # Deprecated aliases for backward compatibility
    'RefereeReportCheckerSummarized',
    'RefereeReportChecker',
]

