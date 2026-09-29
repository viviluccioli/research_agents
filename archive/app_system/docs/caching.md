# Granular Caching System for Referee Reports

## Overview

The referee report system now includes SHA256-based caching with per-round granularity to significantly reduce costs during iterative development and testing. This can save **$1.50-2.00 per run** when working with the same paper and configuration.

## Cost Savings

| Scenario | LLM Calls | Estimated Cost |
|----------|-----------|----------------|
| No cache (first run) | 14-28 | $1.50-2.02 |
| 50% cache hit rate | 7-14 | $0.75-1.01 |
| 80% cache hit rate | 3-6 | $0.30-0.40 |
| 100% cache hit (rerun) | 0 | $0.00 |

## How It Works

### Cache Key Generation

The cache key is computed from:
- **Paper text** (full content SHA256)
- **Selected personas** (sorted for determinism)
- **Persona weights** (sorted by persona name)
- **Model name** (e.g., `claude-3-7-sonnet`)
- **Paper type** (empirical/theoretical/policy)
- **Custom evaluation context**

**Excluded from cache key** (don't affect results):
- Timestamps
- Output file paths
- UI settings
- File names

### Granular Round-Level Caching

Each round is cached independently:

```
.referee_cache/
├── {cache_key}/
│   ├── metadata.json           # Paper info, timestamp, model
│   ├── round_0_selection.json  # Persona selection
│   ├── round_1_reports.json    # Independent evaluations
│   ├── round_2a_cross_exam.json # Cross-examination
│   ├── round_2b_answers.json   # Q&A responses
│   ├── round_2c_amendments.json # Final amendments
│   └── round_3_synthesis.json  # Editor decision
```

**Benefits of granular caching:**
- If you only change Round 3 editor prompts, Rounds 0-2C are cached
- If you change a specific persona, only that persona's outputs are recomputed
- Faster iteration during prompt engineering and testing

## Using the Cache

### In the Streamlit UI

1. **Enable/Disable Cache:**
   ```
   ☑️ Use cached results (if available)
   ```
   Default: **ON**

2. **Force Refresh:**
   ```
   ☐ Force refresh (ignore cache)
   ```
   Use this when testing prompt changes or when you want fresh results.

3. **Clear Cache:**
   - **Clear Cache** button: Clears cache for the current paper
   - **Clear Old Cache (>30 days)**: Removes stale entries
   - **Clear All Cache**: Nuclear option (removes all cached results)

4. **Cache Status:**
   The UI shows which rounds are cached:
   ```
   💾 Cache available for: round_1_reports, round_2a_cross_exam
   ```

5. **Cost Savings Display:**
   ```
   💰 Estimated Cost: $0.75 USD (7 LLM calls, 50,000 tokens)
   💾 Cache: 3/6 rounds cached (50% hit rate) — Saved ~$0.75 USD
   ```

### Programmatic Usage

```python
from referee.engine import execute_debate_pipeline

results = await execute_debate_pipeline(
    paper_text=paper_text,
    use_cache=True,           # Enable caching
    force_refresh=False,      # Use cached results if available
    cache_dir=None,           # Use default .referee_cache/
    # ... other parameters
)

# Check cache status in results
cache_info = results['metadata']['cache']
print(f"Cache hit rate: {cache_info['cache_hit_rate']}")
print(f"Cached rounds: {cache_info['cached_rounds']}")
print(f"Savings: ${cache_info['estimated_savings_usd']}")
```

### Cache Management Functions

```python
from referee._utils.cache import (
    compute_cache_key,
    save_round_results,
    load_round_results,
    check_cache_status,
    clear_cache_for_paper,
    clear_cache,
    get_cache_stats
)

# Compute cache key for a paper
cache_key = compute_cache_key(
    paper_text=paper_text,
    selected_personas=["Theorist", "Empiricist", "Historian"],
    weights={"Theorist": 0.4, "Empiricist": 0.35, "Historian": 0.25},
    model_name="claude-3-7-sonnet"
)

# Check what's cached
status = check_cache_status(cache_key)
# Returns: {'round_0_selection': True, 'round_1_reports': True, ...}

# Clear cache for a specific paper
clear_cache_for_paper(cache_key)

# Clear old cache entries (>30 days)
removed, total = clear_cache(older_than_days=30)

# Get cache statistics
stats = get_cache_stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Cache size: {stats['total_size_mb']} MB")
```

## Cache Invalidation

The cache is automatically invalidated when:
- Paper text changes
- Different personas are selected
- Persona weights change
- Model configuration changes
- Paper type changes
- Custom evaluation context changes

**Not invalidated by:**
- Different file names for the same paper
- Different timestamps
- UI settings or display modes

## Thread Safety

The cache uses `filelock` for thread-safe read/write operations. Multiple processes can safely access the cache simultaneously.

## Cache Location

Default: `.referee_cache/` in the `app_system/` directory

To use a custom location:
```python
results = await execute_debate_pipeline(
    paper_text=paper_text,
    cache_dir=Path("/custom/cache/location")
)
```

## Testing

Run the cache test suite:
```bash
cd app_system
python tests/test_cache.py
```

Tests cover:
- Cache key determinism and uniqueness
- Save/load operations
- Metadata handling
- Cache statistics

## Best Practices

1. **Keep cache enabled** during development and testing
2. **Use force refresh** when testing prompt changes
3. **Clear old cache** periodically (>30 days)
4. **Monitor cache size** - each cached paper is ~50-200KB
5. **Back up cache** before major system changes

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Cache key computation | <1ms | SHA256 hash |
| Save round results | ~5ms | JSON serialization + file write |
| Load round results | ~3ms | File read + JSON parse |
| Check cache status | ~1ms | File existence checks |
| Clear cache entry | ~10ms | Directory removal |

## Troubleshooting

### Cache not working

1. Check that `use_cache=True` in the UI or API call
2. Verify `.referee_cache/` directory exists and is writable
3. Check logs for cache-related errors

### Cache key mismatch

If you get unexpected cache misses:
1. Verify paper text hasn't changed (whitespace matters)
2. Check persona selection and weights are identical
3. Ensure model name is consistent

### Cache corruption

If cached results seem wrong:
1. Clear cache for that paper
2. Run with `force_refresh=True`
3. Check `.referee_cache/{cache_key}/metadata.json` for details

### Disk space concerns

Monitor cache size:
```python
from referee._utils.cache import get_cache_stats
stats = get_cache_stats()
print(f"Cache size: {stats['total_size_mb']} MB")
```

Clear old entries:
```python
from referee._utils.cache import clear_cache
removed, total = clear_cache(older_than_days=30)
```

## Future Enhancements

Potential improvements:
- Compression for cached JSON files
- Cache warming (pre-compute common papers)
- Distributed cache support (Redis/Memcached)
- Cache analytics dashboard
- Automatic cache pruning based on size limits
