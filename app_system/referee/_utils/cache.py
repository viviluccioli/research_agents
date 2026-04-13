"""
Granular caching system for the multi-agent debate (MAD) referee report pipeline.

This module provides SHA256-based caching with per-round granularity to significantly
reduce costs during iterative development. Cache keys are computed from paper text,
selected personas, weights, and model configuration.

Cost savings example:
- Without cache: $1.50-2.02 per run
- With cache (50% hit rate): ~$0.75-1.01 per run
- With cache (80% hit rate): ~$0.30-0.40 per run

Cache structure:
    .referee_cache/
    ├── {cache_key}/
    │   ├── metadata.json
    │   ├── round_0_selection.json
    │   ├── round_1_reports.json
    │   ├── round_2a_cross_exam.json
    │   ├── round_2b_answers.json
    │   ├── round_2c_amendments.json
    │   └── round_3_synthesis.json
"""
import json
import hashlib
import os
import datetime
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from filelock import FileLock, Timeout
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Default cache configuration
DEFAULT_CACHE_DIR = Path(".referee_cache")
DEFAULT_CACHE_TTL_DAYS = 30
DEFAULT_CACHE_ENABLED = True

# Round names for caching
CACHE_ROUND_NAMES = {
    0: "round_0_selection",
    1: "round_1_reports",
    "2a": "round_2a_cross_exam",
    "2b": "round_2b_answers",
    "2c": "round_2c_amendments",
    3: "round_3_synthesis"
}


def compute_cache_key(
    paper_text: str,
    selected_personas: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    model_name: Optional[str] = None,
    paper_type: Optional[str] = None,
    custom_context: Optional[str] = None
) -> str:
    """
    Compute a deterministic SHA256 cache key from paper and configuration.

    The cache key includes everything that affects the debate output:
    - Paper text (full content)
    - Selected personas (if known)
    - Persona weights (if known)
    - Model name
    - Paper type guidance
    - Custom evaluation context

    Excludes: timestamps, output paths, UI settings (don't affect results)

    Args:
        paper_text: The full paper text
        selected_personas: List of persona names (optional for Round 0)
        weights: Dict of persona weights (optional for Round 0)
        model_name: Model identifier
        paper_type: Paper type for guidance (empirical/theoretical/policy)
        custom_context: User-provided evaluation priorities

    Returns:
        64-character hex SHA256 hash
    """
    # Build deterministic cache key components
    key_components = [
        f"paper:{paper_text}",
        f"model:{model_name or 'default'}",
        f"paper_type:{paper_type or 'none'}",
        f"custom_context:{custom_context or 'none'}"
    ]

    # Add personas and weights if provided (not available for Round 0)
    if selected_personas:
        # Sort personas for deterministic ordering
        sorted_personas = sorted(selected_personas)
        key_components.append(f"personas:{','.join(sorted_personas)}")

        if weights:
            # Sort by persona name for deterministic ordering
            sorted_weights = sorted(weights.items())
            weights_str = ','.join(f"{p}:{w:.6f}" for p, w in sorted_weights)
            key_components.append(f"weights:{weights_str}")

    # Combine and hash
    combined = "\n".join(key_components)
    cache_key = hashlib.sha256(combined.encode('utf-8')).hexdigest()

    return cache_key


def get_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """
    Get the cache directory path, creating it if it doesn't exist.

    Args:
        cache_dir: Optional custom cache directory (defaults to .referee_cache)

    Returns:
        Path to cache directory
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_path(
    cache_key: str,
    cache_dir: Optional[Path] = None
) -> Path:
    """
    Get the cache directory path for a specific cache key.

    Args:
        cache_key: The cache key
        cache_dir: Optional custom cache directory

    Returns:
        Path to cache key directory
    """
    base_dir = get_cache_dir(cache_dir)
    return base_dir / cache_key


def save_round_results(
    cache_key: str,
    round_num: Any,
    results: Dict[str, Any],
    cache_dir: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Save results for a specific round to cache.

    Args:
        cache_key: The cache key
        round_num: Round number (0, 1, "2a", "2b", "2c", 3)
        results: Results dictionary to cache
        cache_dir: Optional custom cache directory
        metadata: Optional metadata to save (only for round 0)

    Returns:
        True if successful, False otherwise
    """
    try:
        cache_path = get_cache_path(cache_key, cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Get round filename
        round_name = CACHE_ROUND_NAMES.get(round_num, f"round_{round_num}")
        cache_file = cache_path / f"{round_name}.json"
        lock_file = cache_path / f"{round_name}.lock"

        # Use file locking for thread safety
        with FileLock(str(lock_file), timeout=10):
            # Save results
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)

            # Save metadata on first round
            if round_num == 0 and metadata:
                metadata_file = cache_path / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

        logger.info(f"Cached round {round_num} results: {cache_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to save cache for round {round_num}: {e}")
        return False


def load_round_results(
    cache_key: str,
    round_num: Any,
    cache_dir: Optional[Path] = None,
    max_age_days: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Load cached results for a specific round.

    Args:
        cache_key: The cache key
        round_num: Round number (0, 1, "2a", "2b", "2c", 3)
        cache_dir: Optional custom cache directory
        max_age_days: Optional maximum age in days (uses TTL if not specified)

    Returns:
        Results dictionary if found and valid, None otherwise
    """
    try:
        cache_path = get_cache_path(cache_key, cache_dir)
        if not cache_path.exists():
            return None

        # Get round filename
        round_name = CACHE_ROUND_NAMES.get(round_num, f"round_{round_num}")
        cache_file = cache_path / f"{round_name}.json"
        lock_file = cache_path / f"{round_name}.lock"

        if not cache_file.exists():
            return None

        # Check cache age
        if max_age_days is None:
            max_age_days = DEFAULT_CACHE_TTL_DAYS

        file_age_days = (datetime.datetime.now().timestamp() - cache_file.stat().st_mtime) / 86400
        if file_age_days > max_age_days:
            logger.info(f"Cache expired for round {round_num} (age: {file_age_days:.1f} days)")
            return None

        # Load with file locking
        with FileLock(str(lock_file), timeout=10):
            with open(cache_file, 'r') as f:
                results = json.load(f)

        logger.info(f"Cache HIT for round {round_num}: {cache_file}")
        return results

    except Timeout:
        logger.warning(f"Timeout waiting for cache lock for round {round_num}")
        return None
    except Exception as e:
        logger.error(f"Failed to load cache for round {round_num}: {e}")
        return None


def check_cache_status(
    cache_key: str,
    cache_dir: Optional[Path] = None
) -> Dict[str, bool]:
    """
    Check which rounds are cached for a given cache key.

    Args:
        cache_key: The cache key
        cache_dir: Optional custom cache directory

    Returns:
        Dictionary mapping round names to cache availability
    """
    cache_path = get_cache_path(cache_key, cache_dir)
    status = {}

    for round_num, round_name in CACHE_ROUND_NAMES.items():
        cache_file = cache_path / f"{round_name}.json"
        status[round_name] = cache_file.exists()

    return status


def clear_cache_for_paper(
    cache_key: str,
    cache_dir: Optional[Path] = None
) -> bool:
    """
    Clear all cached results for a specific paper.

    Args:
        cache_key: The cache key
        cache_dir: Optional custom cache directory

    Returns:
        True if successful, False otherwise
    """
    try:
        cache_path = get_cache_path(cache_key, cache_dir)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            logger.info(f"Cleared cache for key: {cache_key}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return False


def clear_cache(
    older_than_days: Optional[int] = None,
    cache_dir: Optional[Path] = None
) -> Tuple[int, int]:
    """
    Clear cached results older than specified age.

    Args:
        older_than_days: Age threshold in days (default: 30)
        cache_dir: Optional custom cache directory

    Returns:
        Tuple of (number of cache entries removed, total entries scanned)
    """
    if older_than_days is None:
        older_than_days = DEFAULT_CACHE_TTL_DAYS

    base_dir = get_cache_dir(cache_dir)
    removed = 0
    scanned = 0

    try:
        cutoff_time = datetime.datetime.now().timestamp() - (older_than_days * 86400)

        for cache_entry in base_dir.iterdir():
            if not cache_entry.is_dir():
                continue

            scanned += 1

            # Check metadata file age
            metadata_file = cache_entry / "metadata.json"
            if metadata_file.exists():
                if metadata_file.stat().st_mtime < cutoff_time:
                    shutil.rmtree(cache_entry)
                    removed += 1
                    logger.info(f"Removed old cache: {cache_entry.name}")

        logger.info(f"Cache cleanup: removed {removed}/{scanned} entries")
        return removed, scanned

    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        return 0, 0


def get_cache_metadata(
    cache_key: str,
    cache_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Load metadata for a cached paper.

    Args:
        cache_key: The cache key
        cache_dir: Optional custom cache directory

    Returns:
        Metadata dictionary if found, None otherwise
    """
    try:
        cache_path = get_cache_path(cache_key, cache_dir)
        metadata_file = cache_path / "metadata.json"

        if not metadata_file.exists():
            return None

        with open(metadata_file, 'r') as f:
            return json.load(f)

    except Exception as e:
        logger.error(f"Failed to load cache metadata: {e}")
        return None


def list_all_cached_papers(
    cache_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    List all cached papers with their metadata.

    Args:
        cache_dir: Optional custom cache directory

    Returns:
        List of metadata dictionaries
    """
    base_dir = get_cache_dir(cache_dir)
    papers = []

    try:
        for cache_entry in base_dir.iterdir():
            if not cache_entry.is_dir():
                continue

            metadata = get_cache_metadata(cache_entry.name, cache_dir)
            if metadata:
                metadata['cache_key'] = cache_entry.name
                papers.append(metadata)

        # Sort by timestamp (most recent first)
        papers.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return papers

    except Exception as e:
        logger.error(f"Failed to list cached papers: {e}")
        return []


def get_cache_stats(cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get statistics about the cache.

    Args:
        cache_dir: Optional custom cache directory

    Returns:
        Dictionary with cache statistics
    """
    base_dir = get_cache_dir(cache_dir)

    try:
        total_entries = 0
        total_size_bytes = 0

        for cache_entry in base_dir.iterdir():
            if not cache_entry.is_dir():
                continue

            total_entries += 1

            for file in cache_entry.rglob('*.json'):
                total_size_bytes += file.stat().st_size

        return {
            'total_entries': total_entries,
            'total_size_mb': round(total_size_bytes / (1024 * 1024), 2),
            'cache_dir': str(base_dir.absolute())
        }

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {
            'total_entries': 0,
            'total_size_mb': 0.0,
            'cache_dir': str(base_dir.absolute()),
            'error': str(e)
        }
