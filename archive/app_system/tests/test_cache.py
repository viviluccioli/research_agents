"""
Test script for the referee caching system.

Run this to verify caching works correctly:
    cd app_system
    python tests/test_cache.py
"""
import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from referee._utils.cache import (
    compute_cache_key,
    save_round_results,
    load_round_results,
    check_cache_status,
    clear_cache_for_paper,
    get_cache_stats
)

def test_cache_key_computation():
    """Test that cache keys are deterministic and unique."""
    print("🧪 Test 1: Cache Key Computation")

    paper_text = "This is a test paper about economics."
    personas = ["Theorist", "Empiricist", "Historian"]
    weights = {"Theorist": 0.4, "Empiricist": 0.35, "Historian": 0.25}

    # Same inputs should produce same key
    key1 = compute_cache_key(paper_text, personas, weights, "claude-3-7-sonnet")
    key2 = compute_cache_key(paper_text, personas, weights, "claude-3-7-sonnet")
    assert key1 == key2, "Cache keys should be deterministic"
    print(f"✅ Deterministic key: {key1[:16]}...")

    # Different paper should produce different key
    key3 = compute_cache_key("Different paper text", personas, weights, "claude-3-7-sonnet")
    assert key1 != key3, "Different papers should have different keys"
    print(f"✅ Different paper key: {key3[:16]}...")

    # Different personas should produce different key
    key4 = compute_cache_key(paper_text, ["Theorist", "Visionary"], weights, "claude-3-7-sonnet")
    assert key1 != key4, "Different personas should have different keys"
    print(f"✅ Different personas key: {key4[:16]}...")

    print("✅ Cache key computation tests passed!\n")


def test_cache_save_load():
    """Test saving and loading cached results."""
    print("🧪 Test 2: Save and Load Cache")

    paper_text = "Test paper for cache save/load"
    cache_key = compute_cache_key(paper_text, model_name="test-model")

    # Test data
    round_1_data = {
        "Theorist": "This is a test report from Theorist",
        "Empiricist": "This is a test report from Empiricist",
        "Historian": "This is a test report from Historian"
    }

    # Save Round 1
    success = save_round_results(cache_key, 1, round_1_data)
    assert success, "Should successfully save cache"
    print("✅ Saved Round 1 to cache")

    # Load Round 1
    loaded_data = load_round_results(cache_key, 1)
    assert loaded_data is not None, "Should successfully load cache"
    assert loaded_data == round_1_data, "Loaded data should match saved data"
    print("✅ Loaded Round 1 from cache")

    # Check cache status
    status = check_cache_status(cache_key)
    assert status['round_1_reports'] == True, "Round 1 should be cached"
    assert status['round_2a_cross_exam'] == False, "Round 2A should not be cached"
    print(f"✅ Cache status: {status}")

    # Clean up
    clear_cache_for_paper(cache_key)
    print("✅ Cleaned up test cache\n")


def test_cache_metadata():
    """Test metadata saving and retrieval."""
    print("🧪 Test 3: Cache Metadata")

    paper_text = "Test paper for metadata"
    cache_key = compute_cache_key(paper_text, model_name="test-model")

    # Save Round 0 with metadata
    round_0_data = {
        "selected_personas": ["Theorist", "Empiricist", "Historian"],
        "weights": {"Theorist": 0.4, "Empiricist": 0.35, "Historian": 0.25}
    }

    metadata = {
        "timestamp": "2024-04-13T10:00:00",
        "model": "claude-3-7-sonnet",
        "paper_hash": "abc123"
    }

    save_round_results(cache_key, 0, round_0_data, metadata=metadata)
    print("✅ Saved Round 0 with metadata")

    # Load metadata
    from referee._utils.cache import get_cache_metadata
    loaded_metadata = get_cache_metadata(cache_key)
    assert loaded_metadata is not None, "Should load metadata"
    assert loaded_metadata['model'] == "claude-3-7-sonnet", "Metadata should match"
    print(f"✅ Loaded metadata: {loaded_metadata}")

    # Clean up
    clear_cache_for_paper(cache_key)
    print("✅ Cleaned up test cache\n")


def test_cache_statistics():
    """Test cache statistics."""
    print("🧪 Test 4: Cache Statistics")

    # Get stats
    stats = get_cache_stats()
    print(f"✅ Cache stats: {stats['total_entries']} entries, {stats['total_size_mb']} MB")
    print(f"   Location: {stats['cache_dir']}\n")


def main():
    print("=" * 60)
    print("Testing Referee Caching System")
    print("=" * 60 + "\n")

    try:
        test_cache_key_computation()
        test_cache_save_load()
        test_cache_metadata()
        test_cache_statistics()

        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
