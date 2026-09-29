"""
Internal utilities for the referee package.

This subpackage contains helper functions and utilities that support
the main referee workflow. These are internal implementation details.
"""

from referee._utils.summarizer import summarize_all_rounds
from referee._utils.quote_validator import (
    validate_quotes_in_reports,
    validate_quotes_in_report,
    get_validation_summary,
    mark_unverified_quotes_in_text
)
from referee._utils.cache import (
    compute_cache_key,
    save_round_results,
    load_round_results,
    check_cache_status,
    clear_cache_for_paper,
    clear_cache,
    get_cache_metadata,
    list_all_cached_papers,
    get_cache_stats
)
from referee._utils.deduplicator import (
    deduplicate_findings,
    cluster_similar_findings,
    merge_cluster,
    identify_cross_references,
    get_dedup_config
)

__all__ = [
    'summarize_all_rounds',
    'validate_quotes_in_reports',
    'validate_quotes_in_report',
    'get_validation_summary',
    'mark_unverified_quotes_in_text',
    'compute_cache_key',
    'save_round_results',
    'load_round_results',
    'check_cache_status',
    'clear_cache_for_paper',
    'clear_cache',
    'get_cache_metadata',
    'list_all_cached_papers',
    'get_cache_stats',
    'deduplicate_findings',
    'cluster_similar_findings',
    'merge_cluster',
    'identify_cross_references',
    'get_dedup_config'
]
