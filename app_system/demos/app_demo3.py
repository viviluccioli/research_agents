# app_demo3.py - Demo 3 with Real Evaluation Results
from pathlib import Path
import sys
import datetime

import streamlit as st

APP_SYSTEM_DIR = Path(__file__).resolve().parents[1]
if str(APP_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(APP_SYSTEM_DIR))

from section_eval import SectionEvaluatorApp
from utils import cm

# Page configuration
st.set_page_config(layout="wide", page_title="Evaluation Agent Demo 3", page_icon="📊")

# Custom CSS for styling (includes severity labels from referee.py)
st.markdown("""
<style>
    .persona-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .empiricist-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
    }
    .historian-box {
        background-color: #f3e5f5;
        border-left: 5px solid #7b1fa2;
    }
    .policymaker-box {
        background-color: #fff3e0;
        border-left: 5px solid #f57c00;
    }
    .editor-box {
        background-color: #e8f5e9;
        border-left: 5px solid #388e3c;
    }
    .round-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
    }
    .verdict-pass {
        color: #2e7d32;
        font-weight: bold;
        font-size: 18px;
    }
    .verdict-fail {
        color: #c62828;
        font-weight: bold;
        font-size: 18px;
    }
    .verdict-revise {
        color: #f57c00;
        font-weight: bold;
        font-size: 18px;
    }
    .verdict-reject {
        color: #c62828;
        font-weight: bold;
        font-size: 18px;
    }
    .severity-major {
        color: #b71c1c;
        font-weight: bold;
        background-color: #ffcdd2;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
        margin: 0 4px;
    }
    .severity-minor {
        color: #e65100;
        font-weight: bold;
        background-color: #ffe0b2;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
        margin: 0 4px;
    }
    .severity-fatal {
        color: #ffffff;
        font-weight: bold;
        background-color: #b71c1c;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
        margin: 0 4px;
    }
    .architecture-box {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #ddd;
        margin: 20px 0;
    }
    .concede-box {
        background-color: #fff9c4;
        border-left: 4px solid #f57f17;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .defend-box {
        background-color: #c8e6c9;
        border-left: 4px solid #388e3c;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.title("📊 Research Evaluation Agent - Demo 3")
st.markdown("**Beige Book NLP Paper Evaluation (Real Results from 2026-03-26)**")
st.markdown("---")

# Ensure session state
def _ensure_session_keys():
    """Create session keys that the workflows expect."""
    if "file_data" not in st.session_state:
        st.session_state.file_data = {}
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = None
    if "paper_type" not in st.session_state:
        st.session_state.paper_type = "empirical"

_ensure_session_keys()

# ============================================================================
# SHARED FILE UPLOADER
# ============================================================================
with st.expander("📁 Document Uploader", expanded=True):
    st.markdown("**Upload manuscripts for Section Evaluation** *(Referee Report shows demo data)*")

    uploaded_files = st.file_uploader(
        "Upload your manuscript (PDF, TXT, DOCX, LaTeX)",
        type=['pdf', 'txt', 'docx', 'tex'],
        accept_multiple_files=True,
        key="main_file_uploader"
    )

    if uploaded_files:
        count = 0
        for uploaded_file in uploaded_files:
            st.session_state.file_data[uploaded_file.name] = uploaded_file.getvalue()
            count += 1
        st.success(f"✅ Successfully uploaded {count} file(s)")
        try:
            cm.clear_history()
        except Exception:
            pass

    st.markdown("---")
    st.markdown("**Or paste text directly**")
    paste_format = st.radio(
        "Pasted text format:",
        ["Plain text", "LaTeX source"],
        horizontal=True,
        key="paste_format_radio",
    )
    pasted_text = st.text_area(
        "Paste your manuscript text here (plain text or LaTeX source):",
        height=200,
        key="paste_text_area",
    )
    if st.button("Add pasted text", key="paste_submit_btn"):
        if pasted_text.strip():
            ext = ".tex" if paste_format == "LaTeX source" else ".txt"
            name = f"Pasted Text{ext}"
            st.session_state.file_data[name] = pasted_text.encode("utf-8")
            st.success(f"✅ Added pasted text as **{name}**")
            try:
                cm.clear_history()
            except Exception:
                pass
        else:
            st.warning("Please paste some text first.")

# Show available files
if st.session_state.file_data:
    st.info(f"📄 Uploaded files: {', '.join(list(st.session_state.file_data.keys()))}")

st.markdown("---")

# Paper type selection
st.markdown("### 📋 Select Paper Type")
st.markdown("Choose the type that best matches your manuscript:")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Select Empirical", use_container_width=True, type="primary" if st.session_state.paper_type == "empirical" else "secondary", key="empirical_btn"):
        st.session_state.paper_type = "empirical"

with col2:
    if st.button("Select Theoretical", use_container_width=True, type="primary" if st.session_state.paper_type == "theoretical" else "secondary", key="theoretical_btn"):
        st.session_state.paper_type = "theoretical"

with col3:
    if st.button("Select Policy", use_container_width=True, type="primary" if st.session_state.paper_type == "policy" else "secondary", key="policy_btn"):
        st.session_state.paper_type = "policy"

st.markdown("---")

# Create tabs
tab1, tab2 = st.tabs(["📝 Section Evaluator", "⚖️ Referee Report"])

# ============================================================================
# TAB 1: SECTION EVALUATOR
# ============================================================================
with tab1:
    st.header("Section-by-Section Manuscript Evaluation")

    st.subheader("📐 How It Works")

    st.markdown("**Paper-Type-Aware Evaluation Framework**")

    # Step-by-step process
    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
    <div style="margin-bottom: 20px;">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; font-size: 16px; min-width: 40px; text-align: center;">1</div>
            <div style="margin-left: 15px; color: #333;">
                <strong style="font-size: 16px;">Text Extraction & Section Detection</strong><br/>
                <span style="font-size: 13px;">PDF/LaTeX parsing → Hierarchical section detection</span>
            </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
    <div style="margin-bottom: 20px;">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; font-size: 16px; min-width: 40px; text-align: center;">2</div>
            <div style="margin-left: 15px; color: #333;">
                <strong style="font-size: 16px;">Paper-Type-Specific Criteria Mapping</strong><br/>
                <span style="font-size: 13px;">Different sections and criteria based on paper type</span>
            </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Criteria mapping by paper type
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: white; padding: 15px; border-radius: 8px; border: 3px solid #003d82;">
            <h4 style="color: #003d82; margin-top: 0;">📊 Empirical</h4>
            <p style="font-size: 13px; margin: 8px 0; color: black;"><strong>Key Sections:</strong></p>
            <ul style="font-size: 12px; margin: 0; padding-left: 20px; color: black;">
                <li>Data</li>
                <li>Methodology ⭐ (1.3×)</li>
                <li>Results</li>
                <li>Robustness Checks</li>
            </ul>
            <p style="font-size: 11px; margin-top: 10px; color: #666;">Emphasizes identification strategy and statistical rigor</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: white; padding: 15px; border-radius: 8px; border: 3px solid #1b5e20;">
            <h4 style="color: #1b5e20; margin-top: 0;">📐 Theoretical</h4>
            <p style="font-size: 13px; margin: 8px 0; color: black;"><strong>Key Sections:</strong></p>
            <ul style="font-size: 12px; margin: 0; padding-left: 20px; color: black;">
                <li>Model Setup</li>
                <li>Proofs ⭐ (1.4×)</li>
                <li>Extensions</li>
                <li>Derivations</li>
            </ul>
            <p style="font-size: 11px; margin-top: 10px; color: #666;">Emphasizes mathematical correctness and intuition</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: white; padding: 15px; border-radius: 8px; border: 3px solid #c62828;">
            <h4 style="color: #c62828; margin-top: 0;">🏛️ Policy</h4>
            <p style="font-size: 13px; margin: 8px 0; color: black;"><strong>Key Sections:</strong></p>
            <ul style="font-size: 12px; margin: 0; padding-left: 20px; color: black;">
                <li>Policy Context</li>
                <li>Recommendations ⭐ (1.3×)</li>
                <li>Background</li>
                <li>Analysis</li>
            </ul>
            <p style="font-size: 11px; margin-top: 10px; color: #666;">Emphasizes feasibility and practical applicability</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<p style='text-align: center; font-size: 12px; color: #666; margin-top: 10px;'>⭐ = Highest importance multiplier</p>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
    <div style="margin-bottom: 20px;">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; font-size: 16px; min-width: 40px; text-align: center;">3</div>
            <div style="margin-left: 15px; color: #333;">
                <strong style="font-size: 16px;">Weighted Section Evaluation</strong><br/>
                <span style="font-size: 13px;">Per-criterion scoring (1-5) × Criterion weights → Adjusted Score</span>
            </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
    <div style="margin-bottom: 20px;">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; font-size: 16px; min-width: 40px; text-align: center;">4</div>
            <div style="margin-left: 15px; color: #333;">
                <strong style="font-size: 16px;">Overall Scoring & Feedback</strong><br/>
                <span style="font-size: 13px;">Publication readiness + Actionable improvements per section</span>
            </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Display selected paper type
    st.markdown("### 📄 Paper Type Configuration")
    if st.session_state.paper_type:
        paper_type_display = st.session_state.paper_type.capitalize()
        if st.session_state.paper_type == "empirical":
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 8px; border: 3px solid #003d82; margin: 10px 0;">
                <span style="color: #003d82; font-weight: bold; font-size: 16px;">📊 Paper type selected: {paper_type_display}</span>
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.paper_type == "theoretical":
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 8px; border: 3px solid #1b5e20; margin: 10px 0;">
                <span style="color: #1b5e20; font-weight: bold; font-size: 16px;">📐 Paper type selected: {paper_type_display}</span>
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.paper_type == "policy":
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 8px; border: 3px solid #c62828; margin: 10px 0;">
                <span style="color: #c62828; font-weight: bold; font-size: 16px;">🏛️ Paper type selected: {paper_type_display}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Check if paper type selected and render section evaluator
    if not st.session_state.paper_type:
        st.warning("⚠️ **Please select a paper type above to begin evaluation.**")
    else:
        if st.session_state.active_tab != "Section Evaluator":
            if st.session_state.active_tab is not None:
                try:
                    cm.clear_history()
                except Exception:
                    pass
            st.session_state.active_tab = "Section Evaluator"

        if "section_eval_app" not in st.session_state:
            st.session_state.section_eval_app = SectionEvaluatorApp()

        st.session_state.section_eval_app.render_ui(
            files=st.session_state.get("file_data", {}),
            paper_type=st.session_state.paper_type
        )

# ============================================================================
# TAB 2: REFEREE REPORT (DEMO WITH REAL DATA)
# ============================================================================
with tab2:
    st.header("Multi-Agent Referee Evaluation")

    if st.session_state.active_tab != "Referee Report":
        if st.session_state.active_tab is not None:
            try:
                cm.clear_history()
            except Exception:
                pass
        st.session_state.active_tab = "Referee Report"

    st.markdown("### 🎭 Demo: Real Evaluation Results")
    st.info("**Note:** This shows actual results from the Beige Book NLP paper evaluation (2026-03-26 10:45:58). This demonstrates the multi-agent debate system output format.")

    # Display Multi-Agent Architecture
    st.markdown('<div class="round-header">🏗️ MULTI-AGENT DEBATE ARCHITECTURE</div>', unsafe_allow_html=True)

    with st.expander("📐 **How It Works**", expanded=False):
        st.markdown("#### Multi-Agent Debate + Deterministic Consensus")

        # Step 1
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; font-size: 16px; min-width: 40px; text-align: center;">1</div>
            <div style="margin-left: 20px; color: #333;">
                <strong style="font-size: 18px;">Independent Evaluation</strong><br/>
                <span style="font-size: 14px;">Each agent evaluates independently with domain-specific verdict and severity-labeled findings</span>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Step 2
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; font-size: 16px; min-width: 40px; text-align: center;">2</div>
            <div style="margin-left: 20px; color: #333;">
                <strong style="font-size: 18px;">Cross-Examination & Debate</strong><br/>
                <span style="font-size: 14px;">Agents challenge each other's reasoning, answer questions, and synthesize perspectives</span>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Step 3
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; font-size: 16px; min-width: 40px; text-align: center;">3</div>
            <div style="margin-left: 20px; color: #333;">
                <strong style="font-size: 18px;">Deterministic Consensus Calculation</strong><br/>
                <span style="font-size: 14px;">Weighted score (PASS=1.0, REVISE=0.5, FAIL=0.0) → Decision thresholds → Final verdict</span>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Round 0: Persona Selection
    st.markdown('<div class="round-header">📋 ROUND 0: PERSONA SELECTION</div>', unsafe_allow_html=True)

    with st.expander("🤖 View System Prompt for Round 0", expanded=False):
        st.code("""You are the Chief Editor of an economics journal. You must select exactly THREE expert personas to review the provided paper.
The available personas are:
1. "Theorist": Focuses on formal mathematical proofs, logic, and model insight.
2. "Empiricist": Focuses on data, econometrics, identification strategy, and statistical validity.
3. "Historian": Focuses on literature lineage, historical background, and appropriate situating of the paper in relevant context.
4. "Visionary": Focuses on novelty and intellectual impact.
5. "Policymaker": Focuses on real-world application, welfare implications, and policy relevance.

Select the 3 most crucial personas for reviewing this specific paper. Assign them weights based on their relative importance to assessing THIS SPECIFIC PAPER. The weights must sum exactly to 1.0.

OUTPUT FORMAT: Return ONLY a valid JSON object. No markdown formatting, no explanations.
{
  "selected_personas": ["Persona1", "Persona2", "Persona3"],
  "weights": {
    "Persona1": 0.4,
    "Persona2": 0.35,
    "Persona3": 0.25
  },
  "justification": "1 sentence explaining the choice and weights."
}""", language="text")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 10px 0;">
        <h4 style="margin-top: 0; color: white;">📊 Selected Panel</h4>
        <p style="margin: 5px 0;"><strong>Empiricist</strong> (Weight: 0.5)</p>
        <p style="margin: 5px 0;"><strong>Policymaker</strong> (Weight: 0.3)</p>
        <p style="margin: 5px 0;"><strong>Historian</strong> (Weight: 0.2)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Justification:** This applied econometric paper using NLP on Beige Book data requires rigorous evaluation of identification strategy and robustness (Empiricist), assessment of practical policy relevance for Fed forecasting (Policymaker), and proper contextualization within the textual analysis and central banking literature (Historian).
    """)

    st.markdown("---")

    # Round 1: Independent Evaluation
    st.markdown('<div class="round-header">📝 ROUND 1: INDEPENDENT EVALUATION</div>', unsafe_allow_html=True)
    st.markdown("*Each persona independently evaluates the paper with severity-weighted findings.*")

    with st.expander("🤖 View System Prompts for Round 1", expanded=False):
        st.markdown("### Error Severity Guide (Applied to All Personas)")
        st.code("""### ERROR SEVERITY — MANDATORY CLASSIFICATION
For every flaw you identify, label it as one of:
- **[FATAL]** — Invalidates core claims. A single FATAL flaw alone justifies FAIL.
  Examples: broken exclusion restriction, mathematical error voiding a key proof,
  data that cannot support the causal claim, fabricated evidence.
- **[MAJOR]** — Requires substantial revision; does not auto-justify FAIL unless multiple co-exist.
  Examples: missing robustness checks on the main result, unaddressed alternative explanations,
  proofs with unverified boundary conditions affecting generality.
- **[MINOR]** — Improves the paper but does not block publication.
  Examples: missing citation, incidental typo, an additional robustness check that would
  strengthen but not reverse a finding.

Your **Verdict** must be consistent with your severity labels:
- Any [FATAL] flaw → FAIL (unless you can explicitly justify why it is non-central)
- Two or more [MAJOR] flaws → REVISE
- Only [MINOR] flaws → PASS""", language="text")

        st.markdown("---")
        st.markdown("### 📊 Empiricist System Prompt")
        st.code("""### ROLE
You are a rigorous Econometrician. You focus on data structures, identification strategies, and statistical validity. You appreciate novel theory and policy relevance, but bad data poisons good ideas.

### OBJECTIVE
1. Empirical Validity: Does the model fit the data? Are standard errors clustered correctly? Is endogeneity addressed? Are empirical decisions explained well?
2. Proportional Error Weighting: Contextualize errors using the classification below. A minor robustness check failing should not sink a paper if the core identification strategy is sound.

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **Empirical Audit**: [Brief overview of your assessment]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote, table number, or figure reference from paper"
- **Verdict**: [PASS/REVISE/FAIL — must be consistent with severity labels above]

CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section.""", language="text")

        st.markdown("---")
        st.markdown("### 🏛️ Policymaker System Prompt")
        st.code("""### ROLE
You are a Senior Policy Advisor (e.g., at the Federal Reserve). You care about policy applicability, welfare implications, and actionable insights from this paper. You rely on your peers for technical accuracy, but you ask: "So what?"

### OBJECTIVE
1. Policy Relevance: Can a central bank, government, and/or think tank/research institution use this to make better policy recommendations and decisions?
2. Practical Translation: Does the paper translate its academic findings into clear, usable implications for the real world?

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **Policy Applicability**: [How can policymakers use this?]
- **Welfare Implications**: [Real-world impact assessment]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote demonstrating policy relevance from paper"
- **Verdict**: [PASS/REVISE/FAIL — must be consistent with severity labels above]

CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section.""", language="text")

        st.markdown("---")
        st.markdown("### 📜 Historian System Prompt")
        st.code("""### ROLE
You are an Economic Historian. You focus on literature lineage and context. You appreciate theoretical and empirical advancements, but above all, you despise researchers who claim to fill a gap in the literature that does not exist/is unfounded.

### OBJECTIVE
1. Contextualization: What literature does this build on?
2. Differentiation: Is the gap presented real, and do they fill it convincingly?

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **Lineage & Context**: [Identify key predecessors]
- **Gap Analysis**: [Assess the claimed gap]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote from paper"
- **Verdict**: [PASS/REVISE/FAIL — must be consistent with severity labels above]

CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section.""", language="text")

    # Empiricist Round 1
    st.markdown('<div class="persona-box empiricist-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Empiricist")
    st.markdown("""
    **Empirical Audit:** This paper applies FinBERT sentiment analysis to Beige Book text (1970-2025) to nowcast/forecast GDP growth and recessions, and to analyze regional economic activity. The core empirical strategy—regressing economic outcomes on sentiment scores while controlling for standard predictors—is straightforward. However, the paper suffers from **critical identification problems**, **inadequate robustness testing**, and **insufficient justification of key methodological choices** that undermine confidence in the causal claims.

    **Key Findings:**
    """)

    st.markdown(f"""
    <span class="severity-fatal">⚠️ FATAL</span> **Reverse causality not addressed:** Current economic conditions likely influence what business contacts report to Fed Banks, making Beige Book sentiment endogenous to contemporaneous GDP.

    <span class="severity-major">🔴 MAJOR</span> **FinBERT sentiment lacks validation:** No validation exercise, inter-rater reliability check, or comparison with human-coded sentiment is provided for the Beige Book corpus.

    <span class="severity-major">🔴 MAJOR</span> **Quarterly aggregation arbitrary:** The "last observation in the quarter" method is unjustified, potentially inducing measurement error.

    <span class="severity-major">🔴 MAJOR</span> **No correction for multiple testing:** 24 total specifications across Tables 1-6 without multiple comparison adjustment.

    <span class="severity-minor">🟠 MINOR</span> **Conservative writing style bias:** Paper acknowledges negative sentiment bias but doesn't address whether this bias is time-varying.
    """, unsafe_allow_html=True)

    st.markdown('<p class="verdict-fail"><strong>Verdict:</strong> FAIL</p>', unsafe_allow_html=True)
    st.markdown("""
    **Justification:** The paper contains **one FATAL flaw**—failure to address reverse causality in the nowcasting specifications—which invalidates the core causal claim. Without an instrument or exogenous shock, the coefficients cannot distinguish whether sentiment *predicts* GDP or merely *reflects* it. Multiple MAJOR flaws compound the problem: lack of sentiment measure validation, omitted variable bias, inadequate standard error adjustments, and opaque regional data construction.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Policymaker Round 1
    st.markdown('<div class="persona-box policymaker-box">', unsafe_allow_html=True)
    st.markdown("### 🏛️ Policymaker")
    st.markdown("""
    **Policy Applicability:** This paper provides policymakers with three actionable tools: (1) Real-time recession early warning system, (2) Regional economic monitoring with spillover effects, (3) Contextual intelligence for crisis management through topic modeling. However, the paper demonstrates statistical significance without policy counterfactuals or decision-theoretic framework.

    **Key Findings:**
    """)

    st.markdown(f"""
    <span class="severity-major">🔴 MAJOR</span> **No policy counterfactuals:** Paper shows Beige Book *predicts* recessions better than alternatives, but never demonstrates that *acting* on this information would have led to superior policy outcomes.

    <span class="severity-major">🔴 MAJOR</span> **Weak out-of-sample forecasting:** Table 3 shows mostly insignificant improvements in GDP forecasts when competing with professional forecasts and news sentiment.

    <span class="severity-major">🔴 MAJOR</span> **Regional analysis lacks policy framework:** Provides no guidance on how policymakers should respond to geographic heterogeneity.

    <span class="severity-minor">🟠 MINOR</span> **Topic modeling descriptive, not predictive:** LDA analysis reads more like historical commentary than a policy tool.
    """, unsafe_allow_html=True)

    st.markdown('<p class="verdict-revise"><strong>Verdict:</strong> REVISE</p>', unsafe_allow_html=True)
    st.markdown("""
    **Justification:** The paper contains three [MAJOR] flaws that substantially limit policy applicability: no policy counterfactual analysis, weak out-of-sample GDP performance, and regional analysis without actionable framework. For policy relevance, the paper needs historical policy simulations, explicit decision rules, and framework for how regional information should influence national monetary policy.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Historian Round 1
    st.markdown('<div class="persona-box historian-box">', unsafe_allow_html=True)
    st.markdown("### 📜 Historian")
    st.markdown("""
    **Lineage & Context:** This paper positions itself within the literature on extracting quantitative signals from qualitative text, building on Balke & Petersen (2002), Balke, Fulmer & Zhang (2017), Sadique et al. (2013), Gascon & Werner (2022), Burke & Nelson (2025), and Araci's FinBERT (2019). The paper applies modern NLP (FinBERT) to the full 1970-2025 Beige Book corpus, extending to regional panels and topic modeling.

    **Key Findings:**
    """)

    st.markdown(f"""
    <span class="severity-major">🔴 MAJOR</span> **FinBERT negative bias not addressed:** Systematic negative bias due to "conservative writing style" threatens construct validity across all specifications.

    <span class="severity-major">🔴 MAJOR</span> **Asymmetric COVID treatment:** COVID excluded as "unique" for estimation but used to validate out-of-sample performance (2020 onward).

    <span class="severity-major">🔴 MAJOR</span> **Endogeneity not addressed:** Beige Book contacts observe current conditions—sentiment may be better contemporaneous measure rather than providing independent predictive information.

    <span class="severity-minor">🟠 MINOR</span> **"Modern NLP vs. dictionary" distinction overstated:** Paper doesn't establish why FinBERT should theoretically outperform domain-specific dictionaries.
    """, unsafe_allow_html=True)

    st.markdown('<p class="verdict-revise"><strong>Verdict:</strong> REVISE</p>', unsafe_allow_html=True)
    st.markdown("""
    **Justification:** The paper contains **no fatal flaws**—core empirical results appear sound within their scope. However, **multiple major issues** require substantial revision: systematic negative bias needs validation/recalibration, endogeneity concerns must be addressed, multiple testing needs correction, and COVID treatment needs justification or correction. The paper makes a solid incremental contribution but methodological rigor needs strengthening.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Round 2A: Cross-Examination Questions (3-column layout)
    st.markdown('<div class="round-header">🔍 ROUND 2A: CROSS-EXAMINATION</div>', unsafe_allow_html=True)
    st.markdown("*Personas challenge each other's reasoning and request clarification on key issues.*")

    with st.expander("🤖 View System Prompt for Round 2A", expanded=False):
        st.code("""### CONTEXT
You are the {role}. You have read the Round 1 evaluations from your peers:
- {peer_1_role} Report: {peer_1_report}
- {peer_2_role} Report: {peer_2_report}

### OBJECTIVE
Engage in cross-domain examination. You respect their domains and want to synthesize perspectives to collectively find the objective truth THROUGH DEBATE. If a peer praised something your domain proves flawed, push back and point it out.

### OUTPUT FORMAT (MANDATORY - ALL THREE SECTIONS REQUIRED)
You MUST include all three sections below. Do not skip any section.

**Cross-Domain Insights**:
[Write 1 concise paragraph synthesizing how their Round 1 evaluations relate to your domain. Do they validate your concerns? Do they miss something crucial from your perspective? Focus on synthesis, not repetition.]

**Constructive Pushback**:
[Write 1 concise paragraph identifying specific conflicts between their assessments and your domain expertise. If a peer gave PASS but you see a flaw, explain why. If a peer gave FAIL but you think it's salvageable, explain why. Be specific about which findings you agree/disagree with.]

**Clarification Requests**:
To {peer_1_role}: [Ask ONE specific, answerable question about their Round 1 evaluation]
To {peer_2_role}: [Ask ONE specific, answerable question about their Round 1 evaluation]

CRITICAL: Do NOT re-list your findings with severity labels here. That was Round 1. This round is about synthesis and dialogue.""", language="text")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="persona-box empiricist-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Empiricist")
        st.markdown("**Questions to Policymaker:**")
        st.markdown("""
        1. You classify "no policy counterfactuals" as [MAJOR] rather than [FATAL]. If the paper cannot establish that *acting* on Beige Book sentiment would improve policy outcomes, how can policymakers justify using this tool? Doesn't this identification failure warrant a stronger severity label?
        """)
        st.markdown("**Questions to Historian:**")
        st.markdown("""
        2. You assign "FinBERT negative bias not addressed" as [MAJOR] rather than [FATAL]. If the sentiment measure is systematically biased and uncalibrated for this corpus, doesn't that invalidate all specifications that rely on it as the key explanatory variable?
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="persona-box policymaker-box">', unsafe_allow_html=True)
        st.markdown("### 🏛️ Policymaker")
        st.markdown("**Questions to Empiricist:**")
        st.markdown("""
        1. Your [FATAL] classification of reverse causality is persuasive for *nowcasting* specifications, but does it apply equally to *forecasting* specifications using lagged regressors (Table 5)? If sentiment from quarter t-1 predicts outcomes in quarter t, isn't the timing structure sufficient to mitigate reverse causality concerns?
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="persona-box historian-box">', unsafe_allow_html=True)
        st.markdown("### 📜 Historian")
        st.markdown("**Questions to Empiricist:**")
        st.markdown("""
        1. You classify "no correction for multiple testing" as [MAJOR]. How many independent hypotheses are actually being tested? If Tables 1-6 all evaluate the same core hypothesis (Beige Book sentiment predicts economic outcomes) using different specifications and samples, is multiple testing correction necessary or would it be overly conservative?
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Round 2B: Responses (3-column layout)
    st.markdown('<div class="round-header">💬 ROUND 2B: RESPONSES</div>', unsafe_allow_html=True)
    st.markdown("*Personas defend their positions or concede points based on evidence.*")

    with st.expander("🤖 View System Prompt for Round 2B", expanded=False):
        st.code("""### CONTEXT
You are the {role}. In the previous round, your peers cross-examined the panel.
Here is the transcript of their cross-examinations:
{r2a_transcript}

### OBJECTIVE
Read the transcript carefully. Identify the specific questions directed AT YOU by your peers. Answer them directly, providing context and TEXTUAL EVIDENCE to address the concerns.

### OUTPUT FORMAT (STRICT)
For EACH peer, provide:

**Response to {peer_1_role}**:
[Your detailed answer with evidence]
**Position**: [CONCEDE or DEFEND] - [One sentence explaining why]

**Response to {peer_2_role}**:
[Your detailed answer with evidence]
**Position**: [CONCEDE or DEFEND] - [One sentence explaining why]""", language="text")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="persona-box empiricist-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Empiricist")

        st.markdown('<div class="defend-box">', unsafe_allow_html=True)
        st.markdown("**Response to Policymaker:**")
        st.markdown("""
        **Defend:** You are correct that reverse causality is mitigated in the *forecasting* specifications (Table 5). However, the paper's *lead contribution*—the claim in the abstract and introduction—is about nowcasting, not forecasting. The core identification failure remains fatal to the paper's primary claim.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="concede-box">', unsafe_allow_html=True)
        st.markdown("**Response to Historian:**")
        st.markdown("""
        **Concede:** You raise a fair point about multiple testing. If all specifications test the same core hypothesis using different samples/controls, then Bonferroni correction may be too strict. I'll downgrade this from [MAJOR] to [MINOR].
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="persona-box policymaker-box">', unsafe_allow_html=True)
        st.markdown("### 🏛️ Policymaker")

        st.markdown('<div class="concede-box">', unsafe_allow_html=True)
        st.markdown("**Response to Empiricist:**")
        st.markdown("""
        **Concede:** Your reverse causality argument is compelling. If the paper cannot establish *direction* of information flow (does sentiment predict or reflect?), then my "no policy counterfactuals" critique is understating the problem. I'll upgrade this to [FATAL] and revise my verdict from REVISE to FAIL.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="persona-box historian-box">', unsafe_allow_html=True)
        st.markdown("### 📜 Historian")

        st.markdown('<div class="concede-box">', unsafe_allow_html=True)
        st.markdown("**Response to Empiricist:**")
        st.markdown("""
        **Concede:** You are correct that the systematic negative bias is more than a measurement concern—it's a construct validity problem. After considering your cross-examination and the Policymaker's literature gap point, I now believe the absence of *any* validation makes all results questionable. I'll strengthen my verdict from REVISE to FAIL based on the cumulative weight of flaws.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Round 2C: Final Amendments (condensed)
    st.markdown('<div class="round-header">📋 ROUND 2C: FINAL AMENDMENTS</div>', unsafe_allow_html=True)
    st.markdown("*After debate and cross-examination, personas revise their verdicts.*")

    with st.expander("🤖 View System Prompt for Round 2C", expanded=False):
        st.code("""### CONTEXT
You are the {role}. The debate is now complete. You have seen:
- Your initial Round 1 evaluation
- Your peers' Round 1 evaluations
- Cross-examination questions and challenges (Round 2A)
- Direct answers to those questions (Round 2B)

Here is the full debate transcript:
{debate_transcript}

### OBJECTIVE
Submit your **Final Amended Verdict** after integrating all debate information. You must:
1. Identify which peer critiques were valid and changed your assessment
2. Explain which of your original concerns remain or were resolved
3. Provide a final verdict (PASS/REVISE/FAIL) with clear justification

### RULES
- Be specific: cite which peer's argument changed your view (if any)
- If you're conceding a point, explain why the peer's evidence was compelling
- If you're defending your original position, explain why peer critiques don't change your assessment
- Your final verdict may be the same as Round 1 OR different - either is acceptable if justified

### OUTPUT FORMAT (MANDATORY - DO NOT SKIP ANY SECTION)
**Insights Absorbed:**
[2-3 sentences: Which peer arguments were most compelling? What new information did you gain from the debate?]

**Changes to Original Assessment:**
[2-3 sentences: Did your verdict change? If yes, why? If no, why do you maintain your original position despite peer challenges?]

**Final Verdict:** [PASS / REVISE / FAIL - you MUST choose one]

**Final Rationale:**
[3-4 sentences justifying your final verdict, explicitly referencing the debate. Include specific concerns that must be addressed if REVISE or FAIL.]

CRITICAL: You MUST provide all four sections above. Do not skip any section. Your final verdict must be one of: PASS, REVISE, or FAIL.""", language="text")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="persona-box empiricist-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Empiricist")
        st.markdown("""
        **Changes:** Maintained FAIL verdict but refined justification. The Historian's literature gap argument (Burke & Nelson 2025 already demonstrates district-level forecasting) revealed even strongest results may be redundant.
        """)
        st.markdown('<p class="verdict-fail"><strong>Final Verdict:</strong> FAIL</p>', unsafe_allow_html=True)
        st.markdown("""
        **Rationale:** Three compounding [FATAL] flaws: (1) Reverse causality in nowcasting invalidates lead claim, (2) Absence of methodological validation means contribution unclear, (3) COVID treatment contradiction undermines credibility. Substantial reconceptualization required.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="persona-box policymaker-box">', unsafe_allow_html=True)
        st.markdown("### 🏛️ Policymaker")
        st.markdown("""
        **Changes:** Changed from REVISE to FAIL. The Empiricist's identification flaw isn't just about missing policy counterfactuals—paper cannot establish *direction* of information flow between sentiment and outcomes.
        """)
        st.markdown('<p class="verdict-fail"><strong>Final Verdict:</strong> FAIL</p>', unsafe_allow_html=True)
        st.markdown("""
        **Rationale:** Three fatal flaws survived debate: (1) Identification failure in nowcasting, (2) Literature gap—no demonstration that FinBERT adds value beyond Burke & Nelson (2025), (3) COVID exclusion contradiction undermines policy credibility.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="persona-box historian-box">', unsafe_allow_html=True)
        st.markdown("### 📜 Historian")
        st.markdown("""
        **Changes:** Changed from REVISE to FAIL. Both reviewers converged on point I underweighted: paper never demonstrates FinBERT adds value beyond existing literature, making claimed "gap" potentially non-existent.
        """)
        st.markdown('<p class="verdict-fail"><strong>Final Verdict:</strong> FAIL</p>', unsafe_allow_html=True)
        st.markdown("""
        **Rationale:** Three compounding fatal flaws: (1) Reverse causality invalidates nowcasting claims, (2) Literature gap—no methodological benchmarking against existing methods, (3) COVID circularity undermines policy relevance.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Round 3: Consensus & Editor Decision
    st.markdown('<div class="round-header">🏛️ ROUND 3: WEIGHTED CONSENSUS & EDITOR DECISION</div>', unsafe_allow_html=True)

    with st.expander("🤖 View System Prompt for Round 3 (Editor)", expanded=False):
        st.code("""### ROLE
You are the Senior Editor writing the final decision letter to the authors.

### COMPUTED MATHEMATICAL CONSENSUS (FIXED - DO NOT RECALCULATE)
The weighted consensus has been deterministically computed:
- **Weighted Consensus Score**: {computed_score} (out of 1.0)
- **Final Decision**: {computed_decision}

### CALCULATION DETAILS
The score was computed using the endogenous weighting system:
- Verdict values: PASS = 1.0, REVISE = 0.5, FAIL/REJECT = 0.0
- Decision thresholds: >0.75 = ACCEPT, 0.40-0.75 = REJECT AND RESUBMIT, <0.40 = REJECT

Individual contributions:
{calculation_details}

### PANEL CONTEXT & WEIGHTS
The following personas were selected for this paper, with these specific weights:
{weights_json}

### AMENDED REPORTS FROM PANEL
{final_reports_text}

### YOUR TASK
Write an official referee letter that synthesizes the panel's evaluation. The decision is FIXED at "{computed_decision}" based on the mathematical consensus. Your job is to write a professional letter justifying this decision.

CRITICAL FORMATTING REQUIREMENTS:
- Write the **Official Referee Report** as a LETTER (prose paragraphs), NOT bullet points
- Draw from panel findings with textual evidence where available
- Maintain a formal, professional tone appropriate for an academic journal
- Detail required fixes or reasons for rejection in narrative form

### OUTPUT FORMAT
**Weight Calculation:**
[Show the explicit math: For each persona, state their verdict (PASS/REVISE/FAIL), their weight, and their weighted contribution. Then show the sum and how it maps to the decision threshold.]

**Debate Synthesis:**
[2-3 sentences summarizing how the panel's views aligned or diverged, and what the final consensus represents]

**Final Decision:** {computed_decision}

**Official Referee Report:**
[Write a formal letter to the authors. Begin with a brief overview of the evaluation process. Then discuss the panel's assessment in paragraph form, covering both strengths and concerns. If REJECT or REJECT AND RESUBMIT, detail the critical issues that led to this decision with evidence from the panel reports. Conclude with clear guidance on next steps. Use complete paragraphs, not bullet points.]""", language="text")

    # Display consensus calculation
    st.markdown("### 🔢 Deterministic Weighted Consensus")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Individual Verdicts & Weights:**")
        st.markdown("- **Empiricist** (weight=0.5): FAIL = 0.0 → contribution = 0.000")
        st.markdown("- **Policymaker** (weight=0.3): FAIL = 0.0 → contribution = 0.000")
        st.markdown("- **Historian** (weight=0.2): FAIL = 0.0 → contribution = 0.000")

    with col2:
        st.markdown("**Consensus Score:**")
        st.metric("Weighted Score", "0.000", help="Out of 1.0")
        st.markdown("**Decision Threshold:**")
        st.code("""Score > 0.75: ACCEPT
0.40 ≤ Score ≤ 0.75: REJECT & RESUBMIT
Score < 0.40: REJECT""", language="text")

    st.markdown("""
    <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; border: 2px solid #c62828; margin: 15px 0;">
        <strong style="font-size: 18px; color: #c62828;">📊 Computed Decision: REJECT</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Editor's Official Report
    st.markdown('<div class="persona-box editor-box">', unsafe_allow_html=True)
    st.markdown("### 🏛️ Senior Editor's Report")

    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; border: 2px solid #c62828; display: inline-block;">
            <span style="font-size: 24px; font-weight: bold; color: #c62828;">❌ REJECT</span>
        </div>
    </div>
    <p style="text-align: center; color: #666; font-size: 14px;">
    (Decision computed deterministically from weighted consensus)
    </p>
    """, unsafe_allow_html=True)

    st.markdown("**Weight Calculation:**")
    st.markdown("""
    The weighted consensus score was computed as follows:
    - **Empiricist** (weight = 0.5): FAIL = 0.0 → weighted contribution = 0.5 × 0.0 = 0.000
    - **Policymaker** (weight = 0.3): FAIL = 0.0 → weighted contribution = 0.3 × 0.0 = 0.000
    - **Historian** (weight = 0.2): FAIL = 0.0 → weighted contribution = 0.2 × 0.0 = 0.000

    **Weighted Consensus Score** = 0.000 + 0.000 + 0.000 = **0.00**

    Under the decision threshold system (>0.75 = ACCEPT, 0.40-0.75 = REJECT AND RESUBMIT, <0.40 = REJECT), this score yields a decision of **REJECT**.
    """)

    st.markdown("**Debate Synthesis:**")
    st.markdown("""
    The panel achieved unanimous consensus on a REJECT decision, though this convergence emerged through substantive cross-examination rather than initial agreement. The Historian initially recommended REVISE but was persuaded by the Empiricist's identification of reverse causality in nowcasting specifications and the Policymaker's demonstration that the COVID exclusion creates a logical contradiction. All three reviewers ultimately converged on the assessment that the paper suffers from three compounding fatal flaws that collectively invalidate its core contributions and cannot be resolved through revision alone.
    """)

    st.markdown("**Official Referee Report:**")
    st.markdown("""
    Dear Authors,

    Thank you for submitting your manuscript on using FinBERT sentiment analysis of Federal Reserve Beige Book reports to nowcast and forecast macroeconomic conditions. Your paper was evaluated by a panel of three expert reviewers representing empirical, policy-oriented, and historical-contextual perspectives, weighted according to the paper's methodological and substantive focus. After thorough deliberation and cross-examination, the panel reached unanimous consensus that the manuscript cannot be recommended for publication in its current form.

    The panel acknowledges several strengths in your work. The application of FinBERT to Federal Reserve Beige Book text represents a methodologically current approach to sentiment analysis, and your recession forecasting results using lagged regressors (Table 5) demonstrate statistical significance. The paper addresses a policy-relevant question about whether qualitative Federal Reserve assessments contain information useful for macroeconomic prediction, and your district-level analysis provides granular empirical evidence.

    However, the panel identified three fundamental and compounding flaws that collectively invalidate the paper's core claims. First, your nowcasting specifications suffer from a fatal identification problem that undermines the paper's lead contribution. The contemporaneous regression (Equation 1, Tables 1 and 4) cannot distinguish whether sentiment predicts or merely reflects economic conditions. This reverse causality concern is compounded by your aggregation method: using the "last observation in quarter" approach eliminates any timing advantage the Beige Book might have for real-time forecasting. This makes it impossible to establish the direction of information flow, rendering your central claim—that Beige Book sentiment "provides meaningful explanatory power in nowcasting GDP growth"—empirically unidentifiable.

    Second, the paper fails to demonstrate that FinBERT adds value beyond existing methods in the Beige Book forecasting literature. Burke and Nelson (2025) already demonstrate that district-level Beige Books forecast recessions, yet your paper never compares FinBERT sentiment scores to their approach or to Balke et al. (2017). Without methodological benchmarking, the panel cannot assess whether your recession forecasting results reflect genuine information content or simply replicate existing findings using an unvalidated measurement tool.

    Third, your treatment of the COVID-19 recession creates a logical contradiction that undermines both statistical credibility and policy relevance. You exclude the 2020 recession from your estimation sample, justifying this by arguing it was "too unique," yet you then use out-of-sample performance during this same excluded period to validate your approach's "superior" recession forecasting capability. This circular reasoning is untenable: if COVID-19 was too unusual to inform your model, it cannot serve as evidence of the model's general applicability.

    These three flaws represent fundamental problems in research design, literature positioning, and logical coherence. Addressing them would require: (1) re-specification using instrumental variables or alternative identification approaches to address endogeneity in nowcasting; (2) systematic comparison of FinBERT sentiment measures against existing Beige Book forecasting methods; and (3) either inclusion of COVID-19 in estimation or exclusion from validation claims, with corresponding reconceptualization of scope. The panel concluded that these changes constitute a substantial reconceptualization rather than revision of the existing manuscript.

    We recognize that this decision may be disappointing given the effort invested in this research. However, the panel believes that addressing these fundamental issues would strengthen your contribution substantially. We encourage you to undertake the necessary reconceptualization and consider resubmission as a new manuscript once these core methodological and positioning challenges have been resolved.

    Sincerely,

    The Editorial Team
    """)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Execution Metadata
    st.markdown("### ⚙️ Execution Metadata")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Configuration**")
        st.markdown("""
        - Model: Claude Sonnet 4.5
        - Temperature: 0.7
        - Max Tokens (R0/R1/R2A/R2B): 4096
        - Max Tokens (R2C): 6144
        - Max Tokens (Editor): 8192
        - Thinking Mode: Disabled
        - Retries: 3
        """)

    with col2:
        st.markdown("**Execution Time**")
        st.markdown("""
        - Start: 2026-03-26 10:45:58
        - End: 2026-03-26 10:52:15
        - Total Runtime: 6 minutes 17 seconds
        """)

    st.markdown("---")
    st.caption("📊 Demo 3 - Real evaluation results from Beige Book NLP paper | Generated: 2026-03-26 10:45:58")
