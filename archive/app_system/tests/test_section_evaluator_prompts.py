#!/usr/bin/env python3
"""
Test script for the section evaluator prompt loader system.
Run this to verify that all prompts load correctly.
"""

import sys
from prompts.section_evaluator.prompt_loader import get_section_evaluator_prompt_loader

def test_section_evaluator_prompts():
    """Test that all prompts load successfully."""
    print("=" * 70)
    print("Testing Section Evaluator Prompt Loader")
    print("=" * 70)

    loader = get_section_evaluator_prompt_loader()

    # Test paper type contexts
    print("\n✓ Testing Paper Type Context Prompts:")
    paper_types = ['empirical', 'theoretical', 'policy', 'finance', 'macro', 'systematic_review']
    for paper_type in paper_types:
        try:
            prompt = loader.get_paper_type_context(paper_type)
            print(f"  ✓ {paper_type}: {len(prompt)} characters loaded")
            assert len(prompt) > 50, f"{paper_type} prompt seems too short"
        except Exception as e:
            print(f"  ✗ {paper_type}: FAILED - {e}")
            return False

    # Test section type guidance
    print("\n✓ Testing Section Type Guidance Prompts:")
    section_types = [
        'abstract', 'introduction', 'literature_review', 'data', 'methodology',
        'model_setup', 'proofs', 'extensions', 'results', 'discussion',
        'robustness_checks', 'identification_strategy', 'calibration', 'simulations',
        'stylized_facts', 'policy_context', 'recommendations', 'search_methodology',
        'inclusion_criteria', 'synthesis', 'conclusion', 'background'
    ]
    for section_type in section_types:
        try:
            prompt = loader.get_section_type_guidance(section_type)
            print(f"  ✓ {section_type}: {len(prompt)} characters loaded")
            assert len(prompt) > 50, f"{section_type} prompt seems too short"
        except Exception as e:
            print(f"  ✗ {section_type}: FAILED - {e}")
            return False

    # Test master prompts
    print("\n✓ Testing Master Prompt Templates:")
    master_prompts = [
        'scoring_philosophy',
        'sophistication_assessment',
        'task_instructions',
        'quote_validation'
    ]
    for prompt_name in master_prompts:
        try:
            prompt = loader.get_master_prompt(prompt_name)
            print(f"  ✓ {prompt_name}: {len(prompt)} characters loaded")
            assert len(prompt) > 50, f"{prompt_name} prompt seems too short"
        except Exception as e:
            print(f"  ✗ {prompt_name}: FAILED - {e}")
            return False

    # Test cache
    print("\n✓ Testing Prompt Caching:")
    prompt1 = loader.get_paper_type_context('empirical')
    prompt2 = loader.get_paper_type_context('empirical')
    assert prompt1 is prompt2, "Cache not working - prompts should be same object"
    print("  ✓ Cache working correctly")

    # Test reload
    print("\n✓ Testing Prompt Reload:")
    loader.reload_prompts()
    prompt3 = loader.get_paper_type_context('empirical')
    assert prompt1 == prompt3, "Reload produced different content"
    print("  ✓ Reload working correctly")

    # Test backward compatibility
    print("\n✓ Testing Backward Compatibility:")
    try:
        from section_eval.prompts.templates import (
            PAPER_TYPE_CONTEXTS,
            SECTION_TYPE_PROMPTS,
            QUOTE_VALIDATION_PROMPT,
            build_evaluation_prompt
        )
        print("  ✓ All imports successful")
        print(f"  ✓ PAPER_TYPE_CONTEXTS: {len(PAPER_TYPE_CONTEXTS)} entries")
        print(f"  ✓ SECTION_TYPE_PROMPTS: {len(SECTION_TYPE_PROMPTS)} entries")
        print(f"  ✓ QUOTE_VALIDATION_PROMPT: {len(QUOTE_VALIDATION_PROMPT)} characters")
        print(f"  ✓ build_evaluation_prompt: callable")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
    print("\nSection evaluator prompt loader is working correctly.")
    print("You can now use the section evaluator with external prompts.")
    return True

def show_prompt_sample():
    """Display a sample of a loaded prompt."""
    loader = get_section_evaluator_prompt_loader()
    print("\n" + "=" * 70)
    print("Sample: Empirical Paper Type Context (first 300 characters)")
    print("=" * 70)
    prompt = loader.get_paper_type_context('empirical')
    print(prompt[:300])
    print("...")
    print(f"\n[Total length: {len(prompt)} characters]")

if __name__ == "__main__":
    success = test_section_evaluator_prompts()

    if success and "--show-sample" in sys.argv:
        show_prompt_sample()

    sys.exit(0 if success else 1)
