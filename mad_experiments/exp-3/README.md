# Experiment 3: Standalone MAD with Embedded Prompts

## Overview

This experiment contains a standalone version of the Multi-Agent Debate (MAD) system with **all system prompts embedded directly in the code** rather than loading them from external files.

## Key Differences from Main System

- **Embedded Prompts**: All persona prompts, debate round prompts, error severity guides, and paper type contexts are defined as string constants in the code
- **No External Dependencies**: Does not rely on `prompts/` directory or `prompt_loader.py`
- **Self-Contained**: Can be copied and modified independently without affecting the main system
- **Simplified Version Control**: Prompt changes are tracked directly in git as code changes

## Files

- `madexp3-full.py` - Complete standalone MAD engine with embedded prompts

## Usage

### Basic Usage

```bash
cd mad_experiments/exp-3
python madexp3-full.py path/to/paper.txt
```

### With Options

```bash
# Specify paper type for persona selection guidance
python madexp3-full.py paper.txt --paper-type empirical

# Add custom evaluation context
python madexp3-full.py paper.txt --custom-context "Focus on policy implications"

# Specify output file
python madexp3-full.py paper.txt --output my_results.json
```

### Arguments

- `paper_file` - Path to paper text file (required)
- `--paper-type` - Type of paper: empirical, theoretical, or policy (optional)
- `--custom-context` - Custom evaluation priorities (optional)
- `--output` - Output JSON file (default: debate_results.json)

## Embedded Prompts

The following prompts are embedded in the code:

### Persona System Prompts
- **Theorist** - Focuses on mathematical rigor and formal proofs
- **Empiricist** - Focuses on data, identification, and statistical validity
- **Historian** - Focuses on literature positioning and context
- **Visionary** - Focuses on novelty and intellectual impact
- **Policymaker** - Focuses on real-world applicability and welfare

### Supporting Prompts
- **Error Severity Guide** - Defines FATAL/MAJOR/MINOR classification
- **Paper Type Contexts** - Guidance for empirical/theoretical/policy papers
- **Custom Context Integration** - Instructions for incorporating user priorities
- **Debate Round Prompts** - Templates for R2A, R2B, R2C, and Editor rounds

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "round_0": {
    "selected_personas": [...],
    "weights": {...},
    "justification": "..."
  },
  "round_1": {...},
  "round_2a": {...},
  "round_2b": {...},
  "round_2c": {...},
  "consensus": {
    "verdicts": {...},
    "weighted_score": 0.75,
    "decision": "ACCEPT"
  },
  "final_decision": "...",
  "metadata": {
    "runtime": "...",
    "token_usage": {...},
    "cost_usd": {...}
  }
}
```

## Modifying Prompts

To modify prompts:

1. Edit the relevant string constant at the top of `madexp3-full.py`
2. No need to update config files or versioning - changes are tracked in git
3. Run the script to test your changes

Example:

```python
# Find the prompt you want to modify
THEORIST_PROMPT = """### ROLE
You are a rigorous Economic Theorist...
"""

# Edit it directly
THEORIST_PROMPT = """### ROLE
You are a SUPER rigorous Economic Theorist...
"""
```

## Use Cases

This standalone version is useful for:

- **Experimentation**: Testing prompt variations without affecting the main system
- **Portability**: Sharing a complete MAD implementation in a single file
- **Archiving**: Preserving exact prompt versions for reproducibility
- **Teaching**: Understanding the complete MAD workflow in one place
- **Research**: Conducting controlled experiments with frozen prompt versions

## Dependencies

Requires:
- Python 3.7+
- `app_system/utils.py` for `single_query()` and `count_tokens()`
- API credentials configured in `app_system/.env`

The script automatically adds `app_system/` to the Python path.
