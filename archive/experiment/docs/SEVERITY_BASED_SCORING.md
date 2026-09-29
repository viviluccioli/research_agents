# Severity-Based Scoring System

## Problem with Direct Numeric Scoring

Asking personas to directly assign a 1-10 score is **ambiguous**:
- What does "6" mean vs "7"?
- Different personas may interpret the scale differently
- No clear anchoring points

## Solution: Severity-First, Then Score

**Two-step process:**
1. **First**: Classify issue severity (qualitative labels)
2. **Then**: Assign score within severity range (quantitative)

This provides clear anchoring and reduces arbitrary number selection.

## Severity Classification Framework

### Step 1: Classify Severity Within Your Domain

**CRITICAL (Scores 1-3)**
- Fatal flaws that invalidate core findings
- Fundamental methodological errors
- Cannot be fixed with revision
- Examples: Endogeneity not addressed, invalid instruments, proof has logical error

**MAJOR (Scores 4-5)**
- Significant problems requiring substantial work
- Core methodology needs rework
- Fixable but requires new analysis/data
- Examples: Missing robustness checks, weak identification, incomplete theoretical derivation

**MODERATE (Scores 6-7)**
- Noticeable issues but not fundamental
- Requires moderate revisions
- Improvements needed but framework is sound
- Examples: Additional controls needed, clearer exposition, minor proofs to add

**MINOR (Scores 8-9)**
- Small improvements would strengthen paper
- Mostly presentation/clarity issues
- Framework and execution are solid
- Examples: Notation inconsistencies, missing literature citations, figure improvements

**NONE (Score 10)**
- Exceptional work, publication-ready
- No meaningful improvements needed
- Rare - only for truly outstanding papers

### Step 2: Assign Score Within Severity Range

Once you've classified severity, pick a score within that range based on:
- **Number of issues**: Multiple major issues → lower in range (4), single major issue → higher (5)
- **Ease of fixing**: Hard to fix → lower, easy to address → higher
- **Impact on contribution**: Undermines main result → lower, peripheral → higher

## Updated Prompt Format

### Round 1:
```
**SCORING REQUIREMENT**:
First, classify the overall severity of issues within your domain:
- CRITICAL (1-3): Fatal flaws, cannot be fixed
- MAJOR (4-5): Significant problems, substantial revision needed
- MODERATE (6-7): Noticeable issues, moderate revision needed
- MINOR (8-9): Small improvements, mostly presentation
- NONE (10): Exceptional, publication-ready

Then assign a specific score within that severity range.

**OUTPUT FORMAT**:
- **Verdict**: [PASS/REVISE/FAIL]
- **Severity**: [CRITICAL/MAJOR/MODERATE/MINOR/NONE]
- **Confidence Score**: [X/10] (within severity range above)
- **Rationale**: [Why this severity? Which specific issues led to this score?]
```

### Round 2C:
```
- **Final Verdict**: [PASS / REVISE / FAIL]
- **Final Severity**: [CRITICAL/MAJOR/MODERATE/MINOR/NONE]
- **Final Score**: [X/10] (within severity range)
- **Final Rationale**: [How did debate change severity assessment?]
```

## Benefits

1. **Clear anchoring**: Personas first pick severity bucket, then refine
2. **Consistency**: Severity labels are less ambiguous than raw numbers
3. **Interpretability**: "MAJOR" is clearer than "score 4.5"
4. **Validation**: Can check if score matches severity (score 2 should be CRITICAL)
5. **Cross-persona comparison**: Easier to see if disagreement is on severity level or score within level

## Example Usage

**Example 1: Econometrician**
```
**Verdict**: REVISE
**Severity**: MAJOR
**Confidence Score**: 5/10
**Rationale**: The identification strategy has endogeneity concerns that require 
additional instrumental variables or alternative identification. This is a MAJOR 
issue because it affects core causal claims, but it's fixable with new analysis. 
Score of 5 because single major issue rather than multiple, and pathway to fix 
is clear.
```

**Example 2: Theorist**
```
**Verdict**: FAIL
**Severity**: CRITICAL
**Confidence Score**: 2/10
**Rationale**: Theorem 3's proof has a logical gap in the induction step that 
invalidates the main result. This is CRITICAL because the proof cannot be salvaged 
with minor fixes - the approach itself is flawed. Score of 2 (not 1) because 
Theorems 1-2 are sound, so not everything is wrong.
```

**Example 3: Policymaker**
```
**Verdict**: PASS
**Severity**: MINOR
**Confidence Score**: 8/10
**Rationale**: Policy recommendations are sound and actionable. Only MINOR issues: 
could discuss implementation challenges more and add cost-benefit analysis. Score 
of 8 because these additions would strengthen but aren't essential for publication.
```

## Extraction Updates

Need to update `extract_score_from_report()` to also extract severity:

```python
def extract_severity_from_report(report: str) -> Optional[str]:
    """Extract severity classification from report."""
    patterns = [
        r'\*\*Severity\*\*\s*:+\s*(CRITICAL|MAJOR|MODERATE|MINOR|NONE)',
        r'Severity\s*:+\s*(CRITICAL|MAJOR|MODERATE|MINOR|NONE)',
        r'\*\*Final Severity\*\*\s*:+\s*(CRITICAL|MAJOR|MODERATE|MINOR|NONE)',
        r'Final Severity\s*:+\s*(CRITICAL|MAJOR|MODERATE|MINOR|NONE)',
    ]
    for pattern in patterns:
        match = re.search(pattern, report, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None
```

## Validation Check

After extraction, validate that score aligns with severity:

```python
def validate_score_severity_alignment(score: float, severity: str) -> bool:
    """Check if score falls in expected range for severity."""
    severity_ranges = {
        'CRITICAL': (1, 3),
        'MAJOR': (4, 5),
        'MODERATE': (6, 7),
        'MINOR': (8, 9),
        'NONE': (10, 10)
    }
    
    if severity not in severity_ranges or score is None:
        return False
    
    min_score, max_score = severity_ranges[severity]
    return min_score <= score <= max_score
```

## CSV Columns to Add

- `persona_X_round1_severity`
- `persona_X_final_severity`
- `persona_X_severity_changed` (True if severity level changed R1→R2C)

## Migration Strategy

**Phase 1**: Test severity-based prompts on 3-5 papers
**Phase 2**: Compare reliability vs direct 1-10 scoring
**Phase 3**: If better, switch to severity-based for all future runs

**Backwards compatibility**: Old runs without severity still work (severity columns = None)

## Alternative: Simpler 3-Level Severity

If 5 levels too granular, use simpler version:

- **SEVERE (1-4)**: Major problems, needs substantial revision or reject
- **MODERATE (5-7)**: Issues that need addressing but fixable  
- **MINOR (8-10)**: Small improvements only

Map to verdicts:
- SEVERE → usually FAIL
- MODERATE → usually REVISE
- MINOR → usually PASS

## Expected Improvements

With severity-based scoring:
1. **More consistent** scores across personas
2. **More interpretable** - "MAJOR severity, score 4" clearer than just "4"
3. **Easier to analyze** - can group by severity level
4. **Better for training** - severity labels provide richer signal than single number
5. **Validation built-in** - can flag misaligned severity/score pairs
