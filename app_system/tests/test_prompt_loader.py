#!/usr/bin/env python3
"""
Test script for the prompt loader system.
Run this to verify that all prompts load correctly.
"""

import sys
from prompts.multi_agent_debate.prompt_loader import get_prompt_loader

def test_prompt_loader():
    """Test that all prompts load successfully."""
    print("=" * 60)
    print("Testing Multi-Agent Debate Prompt Loader")
    print("=" * 60)

    loader = get_prompt_loader()

    # Test persona prompts
    print("\n✓ Testing Persona Prompts:")
    personas = ['theorist', 'empiricist', 'historian', 'visionary', 'policymaker']
    for persona in personas:
        try:
            prompt = loader.get_persona_prompt(persona)
            print(f"  ✓ {persona.capitalize()}: {len(prompt)} characters loaded")
            assert "### ROLE" in prompt, f"{persona} prompt missing ROLE section"
            assert "### OBJECTIVE" in prompt, f"{persona} prompt missing OBJECTIVE section"
            assert "### ERROR SEVERITY" in prompt, f"{persona} prompt missing ERROR SEVERITY section"
        except Exception as e:
            print(f"  ✗ {persona.capitalize()}: FAILED - {e}")
            return False

    # Test debate round prompts
    print("\n✓ Testing Debate Round Prompts:")
    rounds = [
        'round_0_selection',
        'round_2a_cross_exam',
        'round_2b_direct_exam',
        'round_2c_final_amendment',
        'round_3_editor'
    ]
    for round_name in rounds:
        try:
            prompt = loader.get_debate_round_prompt(round_name)
            print(f"  ✓ {round_name}: {len(prompt)} characters loaded")
            assert len(prompt) > 100, f"{round_name} prompt seems too short"
        except Exception as e:
            print(f"  ✗ {round_name}: FAILED - {e}")
            return False

    # Test model config
    print("\n✓ Testing Model Configuration:")
    try:
        config = loader.get_model_config()
        print(f"  ✓ Temperature: {config.get('temperature')}")
        print(f"  ✓ Max tokens: {config.get('max_tokens')}")
        print(f"  ✓ Thinking: {config.get('thinking')}")
        print(f"  ✓ Retry: {config.get('retry')}")
    except Exception as e:
        print(f"  ✗ Model config: FAILED - {e}")
        return False

    # Test cache
    print("\n✓ Testing Prompt Caching:")
    prompt1 = loader.get_persona_prompt('theorist')
    prompt2 = loader.get_persona_prompt('theorist')
    assert prompt1 is prompt2, "Cache not working - prompts should be same object"
    print("  ✓ Cache working correctly")

    # Test reload
    print("\n✓ Testing Prompt Reload:")
    loader.reload_prompts()
    prompt3 = loader.get_persona_prompt('theorist')
    assert prompt1 == prompt3, "Reload produced different content"
    print("  ✓ Reload working correctly")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nPrompt loader is working correctly.")
    print("You can now use the multi-agent debate system with external prompts.")
    return True

def show_prompt_sample():
    """Display a sample of a loaded prompt."""
    loader = get_prompt_loader()
    print("\n" + "=" * 60)
    print("Sample: Theorist Prompt (first 500 characters)")
    print("=" * 60)
    prompt = loader.get_persona_prompt('theorist')
    print(prompt[:500])
    print("...")
    print(f"\n[Total length: {len(prompt)} characters]")

if __name__ == "__main__":
    success = test_prompt_loader()

    if success and "--show-sample" in sys.argv:
        show_prompt_sample()

    sys.exit(0 if success else 1)
