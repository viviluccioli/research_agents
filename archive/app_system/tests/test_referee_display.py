#!/usr/bin/env python3
"""
Test referee report display with mock data (no LLM calls).
Run from app_system/: streamlit run test_referee_display.py
"""
import streamlit as st
from referee import RefereeReportChecker

# Mock results with properly formatted Round 2C outputs
MOCK_RESULTS = {
    'round_0': {
        'selected_personas': ['Empiricist', 'Historian', 'Visionary'],
        'weights': {'Empiricist': 0.45, 'Historian': 0.35, 'Visionary': 0.20},
        'justification': 'This paper is primarily empirical with historical context and novel methodology.'
    },
    'round_1': {
        'Empiricist': """**Empirical Audit**: The identification strategy is weak.

[MAJOR] The paper does not adequately address endogeneity concerns in the main specification.
**Source Evidence**: "We use OLS regression with robust standard errors" (page 5)

[MINOR] Missing robustness checks for alternative time periods.
**Source Evidence**: "Our data covers 2000-2020" (page 3)

**Verdict**: REVISE""",
        'Historian': """**Literature Review**: Adequate coverage of recent literature.

[MINOR] Missing citation to seminal work by Smith (2019).
**Source Evidence**: The paper cites Johnson (2020) but not the earlier foundational work.

**Verdict**: PASS""",
        'Visionary': """**Innovation Assessment**: Novel approach to standard problem.

[MAJOR] The methodological contribution is not clearly articulated.
**Source Evidence**: "We employ a standard approach" (page 4) - contradicts claims of novelty.

**Verdict**: REVISE"""
    },
    'round_2a': {
        'Empiricist': """**Cross-Domain Insights**: The Historian raised important points about missing foundational citations that strengthen the identification strategy argument.

**Constructive Pushback**: I challenge the Visionary's claim that this is methodologically novel - the approach appears standard.

**Clarification Requests**: To Visionary: Can you specify which aspects are truly novel versus incremental improvements?""",
        'Historian': """**Cross-Domain Insights**: The Empiricist's endogeneity concerns are well-founded and should be addressed.

**Constructive Pushback**: The identification issues may be more severe than initially assessed.

**Clarification Requests**: To Empiricist: Would instrumental variables address your concerns, or is the problem more fundamental?""",
        'Visionary': """**Cross-Domain Insights**: Both peers raise valid points about the gap between novelty claims and actual contribution.

**Constructive Pushback**: Perhaps I overstated the novelty - this may be more of a solid incremental contribution.

**Clarification Requests**: To Empiricist: If endogeneity is addressed, would this meet publication standards?"""
    },
    'round_2b': {
        'Empiricist': """**Response to Historian**: Yes, instrumental variables could help, but the paper needs stronger theoretical justification for the instruments.

**Response to Visionary**: If endogeneity is properly addressed with valid instruments and robustness checks, yes, this could meet standards.

**Concession or Defense**: I defend my original assessment. The identification strategy requires substantial revision.""",
        'Historian': """**Response to Empiricist**: Agreed on the need for stronger theoretical grounding.

**Response to Visionary**: The novelty claims should be toned down to reflect incremental contribution.

**Concession or Defense**: I concede that my initial PASS was too lenient given the identification issues.""",
        'Visionary': """**Response to Empiricist**: I concede that novelty is overstated.

**Response to Historian**: Agreed that missing citations weaken the positioning.

**Concession or Defense**: I concede my original assessment - this requires revision to address identification and properly frame contribution."""
    },
    'round_2c': {
        'Empiricist': """**Insights Absorbed**: The debate confirmed that identification strategy is the central issue. Both peers agree on its severity.

**Changes to Original Assessment**: I maintain my REVISE verdict. The Historian's concession and Visionary's agreement validate the concerns.

**Final Verdict**: REVISE

**Final Rationale**: The paper requires substantial methodological revision to address endogeneity. While the topic is relevant and literature review adequate, the identification strategy undermines causal claims. Authors must either provide valid instruments with theoretical justification or reframe claims as correlational. The novelty claims should also be tempered to reflect incremental contribution.""",
        'Historian': """**Insights Absorbed**: The empirical concerns are more severe than I initially recognized. My PASS verdict was premature.

**Changes to Original Assessment**: I am changing my verdict from PASS to REVISE based on the fundamental identification issues raised.

**Final Verdict**: REVISE

**Final Rationale**: After debate, I agree with the Empiricist that identification strategy is a major flaw requiring substantial revision. The missing citations are minor by comparison. Authors should address endogeneity concerns before resubmission.""",
        'Visionary': """**Insights Absorbed**: I learned that my initial enthusiasm for novelty was misplaced. The methodological approach is more standard than I recognized.

**Changes to Original Assessment**: I maintain REVISE but for different reasons - now focused on identification rather than novelty framing.

**Final Verdict**: REVISE

**Final Rationale**: The paper needs major revision to address identification strategy and to reframe contribution claims appropriately. The topic remains interesting, but execution requires substantial improvement before publication."""
    },
    'final_decision': """**Weight Calculation**:
Empiricist (weight=0.45): REVISE = 0.5
Historian (weight=0.35): REVISE = 0.5
Visionary (weight=0.20): REVISE = 0.5
Weighted Score = (0.45 × 0.5) + (0.35 × 0.5) + (0.20 × 0.5) = 0.50

**Debate Synthesis**: The panel reached unanimous agreement that the paper requires major revision. All three reviewers converged on REVISE after debate, with identification strategy as the primary concern.

**Final Decision**: REJECT AND RESUBMIT

**Official Referee Report**:

Dear Authors,

After careful multi-agent review, we must reject your manuscript with an invitation to resubmit after major revisions.

**Critical Issues Requiring Revision:**

1. **Identification Strategy**: Your OLS approach does not adequately address endogeneity concerns. You must either provide valid instrumental variables with theoretical justification or reframe your causal claims.

2. **Methodological Framing**: The paper claims methodological novelty that is not substantiated. Please reframe as a solid application of existing methods to a new context.

**Strengths**: The topic is relevant, data coverage is appropriate, and literature review is generally adequate.

We encourage resubmission after addressing these concerns.

Sincerely,
The Editorial Team""",
    'metadata': {
        'model_version': 'anthropic.claude-3-7-sonnet-20250219-v1:0',
        'start_time': '2026-03-26 14:30:00',
        'end_time': '2026-03-26 14:36:27',
        'total_runtime_seconds': 387.0,
        'total_runtime_formatted': '6m 27s',
        'temperature': 0.7,
        'thinking_enabled': False,
        'thinking_budget_tokens': 'N/A',
        'max_tokens': 4096,
        'max_retries': 3,
        'retry_delay_seconds': 5
    }
}

def main():
    st.set_page_config(layout="wide", page_title="Referee Report Display Test")
    st.title("🧪 Referee Report Display Test (Mock Data)")

    st.info("This test uses mock data to verify display logic without making LLM calls. Perfect for testing UI changes!")

    if st.button("Load Mock Results"):
        st.session_state.mock_results = MOCK_RESULTS

    if 'mock_results' in st.session_state:
        checker = RefereeReportChecker()
        checker.display_debate_results(st.session_state.mock_results)

if __name__ == "__main__":
    main()
