#!/usr/bin/env python3
"""
Standalone Streamlit app for Experiment 5 (Severity Delta + Layman Translation).

Run from app_system directory:
    cd app_system
    streamlit run exp_5/app.py

NEW FEATURES IN EXP-5:
1. Severity Delta (Δ) system for flaw categorization
2. Layman Translation requirement for cross-domain understanding
3. Δ-based filtering in Round 2A cross-examination
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from exp_5.workflow import RefereeWorkflowExp5

# Page configuration
st.set_page_config(
    page_title="Experiment 5: Referee System with Severity Delta",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🔬 Experiment 5: Multi-Agent Referee System")
st.caption("**Enhanced with Severity Delta (Δ) and Layman Translation**")

# Explanation banner
with st.expander("ℹ️ What's New in Experiment 5?", expanded=False):
    st.markdown("""
    ### 🎯 New Features

    #### 1. **Severity Delta (Δ) System**
    Personas must categorize flaws using counterfactual reasoning:
    - **Δ-High**: Conclusion breaks if fixed (fatal flaws)
    - **Δ-Medium**: Evidence halved if fixed (significant issues)
    - **Δ-Low**: Presentational only (minor issues)

    #### 2. **Layman Translation**
    Each persona must translate technical critiques into plain English
    for cross-domain understanding.

    #### 3. **Δ-Based Filtering (Round 2A)**
    Personas can only change verdicts based on cross-domain critiques if:
    - Flaw is **Δ-High**, AND
    - Peer's **Confidence Score > 8/10**

    This prevents low-severity cascading errors across domains.

    ---

    ### 📚 Comparison with Base System

    | Feature | Base System | Experiment 5 |
    |---------|------------|--------------|
    | Flaw Categorization | Generic severity labels | Counterfactual Δ system |
    | Cross-Domain Communication | Technical jargon | Layman Translation required |
    | Round 2A Filtering | No filtering | Δ-based filtering (Δ-High + Confidence > 8) |
    | Selection Prompt | Generic descriptions | Value-pillar focused |

    """)

# Sidebar
with st.sidebar:
    st.header("⚙️ Experiment 5 Configuration")

    st.markdown("### 📄 Paper Type")
    paper_type = st.selectbox(
        "Select paper type (guides persona selection)",
        options=["", "empirical", "theoretical", "policy"],
        format_func=lambda x: "Select paper type..." if x == "" else x.title()
    )

    # Store in session state for workflow access
    st.session_state["paper_type"] = paper_type if paper_type else None

    st.markdown("---")
    st.markdown("### 📤 Document Upload")
    st.caption("Upload your manuscript (PDF)")

# File uploader
uploaded_files = st.file_uploader(
    "Upload manuscript (PDF)",
    type=["pdf"],
    accept_multiple_files=False,
    key="doc_uploader"
)

# Convert to dict format expected by workflow
files = {}
if uploaded_files:
    files[uploaded_files.name] = uploaded_files.getvalue()

# Initialize workflow
workflow = RefereeWorkflowExp5()

# Render workflow UI
if files:
    workflow.render_ui(files=files)
else:
    st.info("👆 Please upload a PDF manuscript to begin evaluation.")

    # Show example
    with st.expander("📋 Example Output Format", expanded=False):
        st.markdown("""
        ### Example Persona Report (Round 1)

        **Verdict**: REVISE
        **Score**: 6/10

        #### Structural Strength
        The instrumental variable design is well-motivated and the first-stage F-statistic (23.4) exceeds conventional thresholds.

        #### Domain Audit
        **[MAJOR]** The exclusion restriction is questionable. The instrument (rainfall) likely affects the outcome (agricultural productivity) through channels other than the treatment (technology adoption), violating the exclusion restriction.

        #### Severity Delta (Δ)
        **Δ-High**: If the exclusion restriction violation were fixed (e.g., by finding a true excluded instrument), the causal estimates would fundamentally change. The current instrument confounds multiple causal pathways.

        #### Layman Translation
        The paper tries to measure whether new technology causes farms to produce more, but the tool they use to isolate causation (rainfall as an instrument) also directly affects farm output through other paths, making it impossible to trust the causal claim.

        #### Confidence Score
        9/10 - This is a well-established issue in applied econometrics.

        #### Source Evidence
        > "We use rainfall as an instrument for technology adoption..." (p. 12)
        > Table 3 shows reduced-form effects of rainfall on productivity (p. 18)

        #### Verdict
        REVISE - The identification strategy requires substantial revision.
        """)

# Footer
st.markdown("---")
st.caption("🔬 **Experiment 5** | Severity Delta (Δ) + Layman Translation | No Human-in-the-Loop")
