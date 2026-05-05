# Referee System Design: Workflow and Prompt Architecture

## Overview

The referee system implements a **5-round multi-agent debate (MAD)** to evaluate research papers. The core principle is **adversarial collaboration**: specialized personas independently analyze the paper, engage in structured dialogue to surface disagreements, and reach evidence-based consensus.

## Design Philosophy

**Why multiple personas?** Single LLM evaluation suffers from consistency bias. Multiple independent agents with distinct expertise force explicit articulation of trade-offs (theoretical elegance vs. empirical rigor) that a single evaluator might gloss over.

**Why structured rounds?** Free-form discussion tends toward superficial agreement or circular debate. The 5-round structure enforces progression: independent analysis → targeted questioning → response → reflection → synthesis.

**Why weighted voting?** Not all perspectives are equally relevant. A theoretical paper needs different expertise than an empirical study. The LLM selects 3 personas from 10 and assigns weights based on paper content, allowing adaptive evaluation.

## The 10 Available Personas

The system selects 3 of 10 specialized personas per paper:

**Core Technical**: (1) Theorist: formal models, proofs; (2) Econometrician: causal inference, identification; (3) ML_Expert: model architecture, hyperparameters; (4) Data_Scientist: data pipelines, preprocessing; (5) CS_Expert: algorithms, complexity

**Contextual & Impact**: (6) Historian: literature positioning, originality; (7) Visionary: paradigm-shifting potential; (8) Policymaker: real-world applicability, welfare

**Ethics & Equity**: (9) Ethicist: moral implications, privacy, fairness; (10) Perspective: distributional consequences, algorithmic bias

Each persona has a system prompt defining expertise, output format (summary, strengths, weaknesses, questions, recommendation), error severity taxonomy, and domain-specific guidance. **Key design**: personas are adversarial by default—the Theorist challenges empirical papers, the Econometrician challenges theory, creating productive tension.

## The 5 Rounds

### Round 0: Persona Selection & Weighting

**Purpose**: Match experts to paper content.

**Process**: LLM receives paper abstract + descriptions of all 10 personas → selects 3 → assigns weights summing to 1.0 (e.g., Econometrician: 0.5, Theorist: 0.3, Ethicist: 0.2).

**Design rationale**: 10 persona pool covers technical depth, contextual breadth, and ethics. 3-persona selection balances diversity with efficiency. Weighted voting allows "primary + secondary + tertiary" reviewer model. An empirical ML paper might weight ML_Expert: 0.5, Econometrician: 0.3, Data_Scientist: 0.2; a theoretical paper might weight Theorist: 0.6, Historian: 0.25, Visionary: 0.15.

### Round 1: Independent Analysis

**Context**: Each persona receives full paper + persona system prompt + paper type guidance + Round 1 instructions.

**Execution**: 3 selected personas run in parallel (async). No persona sees others' output.

**Output**: Structured report—summary, strengths (with quotes), weaknesses (with quotes), questions, preliminary recommendation (PASS/REVISE/FAIL).

**Design rationale**: Parallel execution prevents anchoring bias. Quotes ground critiques and enable hallucination detection. Preliminary recommendation forces commitment before discussion.

### Round 2A: Cross-Examination

**Context**: Each persona receives own Round 1 report + peer reports from other 2 personas.

**Execution**: Parallel (async).

**Output**: 3-5 directed questions (e.g., "@Empiricist: You claim the IV is weak, but Table 3 shows F-stat > 10. What threshold are you using?").

**Design rationale**: Directed questions force engagement with specific claims. The "@name" convention structures Q&A for Round 2B.

### Round 2B: Direct Examination

**Context**: Each persona receives own Round 1 report + questions directed at them from Round 2A.

**Execution**: Parallel (async).

**Output**: Structured responses (quote question, then answer with evidence). Must concede when peers identify valid issues or defend when critique stands.

**Design rationale**: Requiring concession or defense prevents non-responsive answers.

### Round 2C: Amendment

**Context**: Each persona receives own Round 1 report + full Q&A transcript (2A + 2B from all personas).

**Execution**: Parallel (async).

**Output**: Updated report (same structure as Round 1, revised). Must explicitly state changes from Round 1.

**Design rationale**: Explicit change tracking makes deliberation visible. Forces engagement with counterarguments.

### Round 3: Editor Synthesis

**Context**: Editor receives full debate transcript (Round 1 + 2A + 2B + 2C) + persona weights.

**Execution**: Single LLM call.

**Consensus formula**: `weighted_score = Σ(persona_weight × persona_score)` where PASS=1.0, REVISE=0.5, FAIL=0.0. If weighted_score > 0.75: ACCEPT; 0.40–0.75: RESUBMIT; < 0.40: REJECT.

**Output**: Referee report—summary judgment, major strengths, major weaknesses, required revisions, optional suggestions, minority dissent.

**Design rationale**: Weighted voting respects differential expertise while preserving contrarian perspectives (minority dissent section). Thresholds (0.75, 0.40) match typical journal accept rates.

## Prompt Composition

Prompts are **composable** and **versioned**:

**System Prompts** (`prompts/multi_agent_debate/personas/{name}/v1.0.txt`): Define persona identity—role, expertise, priorities, error severity guide (injected at runtime via `{error_severity}` placeholder).

**Paper Type Contexts** (`prompts/multi_agent_debate/paper_type_contexts/{type}/v1.0.txt`): Domain-specific guidance injected into persona system prompt (e.g., "Identification strategy is paramount" for empirical papers).

**Round Instructions** (`prompts/multi_agent_debate/debate_rounds/round_{N}/v1.0.txt`): Injected as user message at each round—controls context visibility and output format.

**Composition Example (Round 1)**:
```
[System] {persona_system_prompt} + {paper_type_context} + {error_severity_guide}
[User] {round_1_instructions} + {paper_text}
```

## Design Trade-offs

**Independence vs. Information**: Round 1 personas see only the paper (maximizes independence), but Round 2C provides full visibility after initial commitment (corrects missed issues).

**Structure vs. Flexibility**: Fixed 5-round format ensures comparability but always runs all rounds even if consensus is obvious after Round 1.

**Weights vs. Equal Voice**: Weighted voting better reflects expertise but introduces meta-uncertainty (are the weights correct?). An equal-weight variant would be simpler but less adaptive.

**Consensus vs. Dissent**: Weighted threshold values agreement but minority dissent section preserves contrarian perspectives. Thresholds calibrated empirically via ground truth comparison.

**Cost vs. Quality**: Parallel execution (Rounds 1/2A/2B/2C) increases cost but dramatically improves independence. Sequential execution would be cheaper but personas would anchor on earlier responses.

**10 vs. 5 personas**: Expanded pool allows finer-grained expertise matching. A computational economics paper can now select CS_Expert + Econometrician + Theorist, whereas the old 5-persona system forced awkward compromises. Cost is manageable since we still select only 3 per paper.

## Extending the System

**Adding an 11th persona**: (1) Create `prompts/multi_agent_debate/personas/{name}/v1.0.txt` with `{error_severity}` placeholder; (2) Add to numbered list in `engine.py:SELECTION_PROMPT`; (3) Add to `load_persona_prompt()` persona_dir_map and FALLBACK_SYSTEM_PROMPTS; (4) Add CSS class and icon in `workflow.py`.

**Adding paper types**: (1) Create context file `prompts/.../paper_type_contexts/{type}/v1.0.txt`; (2) Add detection logic; (3) Update persona-specific guidance.

**Modifying rounds**: Edit round instruction files. Ensure appropriate context visibility (e.g., don't give Round 2A the full debate transcript—that comes in Round 2C).

## Quality Assurance

**Quote validation**: Fuzzy match (85-95% threshold) checks quoted text exists in paper. Prevents hallucinated evidence. Results shown in UI.

**Deduplication**: Quote overlap + semantic similarity identifies redundant findings across personas. Merges before display.

**Caching**: SHA256-based per-round caching (cache key: paper text + personas + weights + model). Saves ~50-80% cost during development.

## Implementation Notes

- **Model**: Claude 4.5 Sonnet, temperature 1.0 (thinking mode enabled, 2048 token budget)
- **Retries**: 3× with exponential backoff
- **Execution**: `asyncio.gather()` for parallel rounds
- **Token limits**: Paper text truncated to 100K chars for cache key
- **Output**: Streamlit UI with collapsible sections + Excel export with full transcript
