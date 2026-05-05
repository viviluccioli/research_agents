# Confidence Scoring Rubric (1-10 Scale)

This rubric provides guidance for assigning confidence scores alongside categorical verdicts.

## Score Ranges and Categorical Alignment

- **1-3 = FAIL Zone** (Reject)
  - **1**: Fatal flaws, fundamentally broken
  - **2**: Major flaws, would require complete rewrite
  - **3**: Significant problems, core methodology unsound

- **4-7 = REVISE Zone** (Reject and Resubmit)
  - **4**: Borderline fail, needs substantial revision
  - **5**: Major revisions needed, but salvageable
  - **6**: Moderate revisions across multiple sections
  - **7**: Minor to moderate revisions needed

- **8-10 = PASS Zone** (Accept)
  - **8**: Minor revisions only, nearly ready
  - **9**: Strong paper, minimal improvements needed
  - **10**: Exceptional, publication-ready as is

## How to Assign Scores

**Consider:**
1. **Severity of issues** (fatal vs. minor)
2. **Number of issues** (pervasive vs. isolated)
3. **Ease of revision** (requires new data vs. rewording)
4. **Strength of contribution** (incremental vs. novel)

**Within your domain:**
- Score based on domain-specific criteria
- Be precise: distinguish between 5 (needs major work) and 7 (needs minor fixes)
- Your score reflects confidence in the verdict

## Output Format

**Round 1:**
```
- **Verdict**: [PASS/REVISE/FAIL]
- **Confidence Score**: [X/10]
- **Rationale**: [Why this score?]
```

**Round 2C:**
```
- **Final Verdict**: [PASS/REVISE/FAIL]
- **Final Score**: [X/10]
- **Final Rationale**: [How did debate influence the score?]
```

## Examples

**Example 1: Strong Pass**
- Verdict: PASS
- Score: 9/10
- Rationale: Excellent identification strategy, robust results, minor presentation improvements needed

**Example 2: Borderline Revise/Fail**
- Verdict: REVISE
- Score: 4/10
- Rationale: Identification concerns are addressable but require substantial new analysis

**Example 3: Clear Fail**
- Verdict: FAIL
- Score: 2/10
- Rationale: Fundamental methodological flaws that cannot be fixed with revision
