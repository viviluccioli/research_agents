# app_demo.py - Demo Version with Architecture Diagrams
from pathlib import Path
import sys

import streamlit as st
from PIL import Image

APP_SYSTEM_DIR = Path(__file__).resolve().parents[1]
if str(APP_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(APP_SYSTEM_DIR))

from section_eval import SectionEvaluatorApp
from utils import cm

# Page configuration
st.set_page_config(layout="wide", page_title="Evaluation Agent Demo 2", page_icon="📊")

# Custom CSS for styling
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
    .policy-analyst-box {
        background-color: #fce4ec;
        border-left: 5px solid #c2185b;
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
    .architecture-box {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #ddd;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.title("📊 Research Evaluation Agent - Demo 2")
st.markdown("**AI-powered manuscript evaluation system for economics research**")
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
# SHARED FILE UPLOADER (for Section Evaluator)
# ============================================================================
with st.expander("📁 Document Uploader", expanded=True):
    st.markdown("**Upload manuscripts for Section Evaluation** *(Referee Report uses demo data)*")

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
        # Clear LLM history because content changed
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
    st.info(f"📄 **Available files:** {', '.join(list(st.session_state.file_data.keys()))}")

st.markdown("---")

# ============================================================================
# PAPER TYPE SELECTION (applies to both evaluators)
# ============================================================================
st.markdown("### 📋 Select Paper Type")
st.markdown("*This selection configures evaluation criteria for both Section Evaluator and Referee Report*")

# Create three columns for horizontal layout
col1, col2, col3 = st.columns(3)

# Empirical
with col1:
    st.markdown("""
    <div style="background: white; padding: 18px; border-radius: 8px; border: 3px solid #003d82; margin: 12px 0;">
        <h4 style="color: #003d82; margin: 0 0 8px 0;">📊 Empirical</h4>
        <p style="font-size: 13px; color: black; margin: 0;">
        Uses data to test hypotheses or estimate relationships. Emphasizes identification strategy, statistical rigor, and robustness checks.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Select Empirical", use_container_width=True, type="primary" if st.session_state.paper_type == "empirical" else "secondary", key="empirical_btn"):
        st.session_state.paper_type = "empirical"

# Theoretical
with col2:
    st.markdown("""
    <div style="background: white; padding: 18px; border-radius: 8px; border: 3px solid #1b5e20; margin: 12px 0;">
        <h4 style="color: #1b5e20; margin: 0 0 8px 0;">📐 Theoretical</h4>
        <p style="font-size: 13px; color: black; margin: 0;">
        Develops mathematical models with formal proofs. Emphasizes assumption clarity, mathematical correctness, and economic intuition.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Select Theoretical", use_container_width=True, type="primary" if st.session_state.paper_type == "theoretical" else "secondary", key="theoretical_btn"):
        st.session_state.paper_type = "theoretical"

# Policy
with col3:
    st.markdown("""
    <div style="background: white; padding: 18px; border-radius: 8px; border: 3px solid #c62828; margin: 12px 0;">
        <h4 style="color: #c62828; margin: 0 0 8px 0;">🏛️ Policy</h4>
        <p style="font-size: 13px; color: black; margin: 0;">
        Addresses real-world policy questions with evidence-based recommendations. Emphasizes feasibility, trade-offs, and practical applicability.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Select Policy", use_container_width=True, type="primary" if st.session_state.paper_type == "policy" else "secondary", key="policy_btn"):
        st.session_state.paper_type = "policy"

st.success(f"✓ **Current selection:** {st.session_state.paper_type.capitalize()}")

st.markdown("---")

# Create tabs
tab1, tab2 = st.tabs(["📝 Section Evaluator", "⚖️ Referee Report"])

# ============================================================================
# TAB 1: SECTION EVALUATOR
# ============================================================================
with tab1:
    st.header("Section-by-Section Manuscript Evaluation")

    # Framework Architecture
    # st.markdown('<div class="architecture-box">', unsafe_allow_html=True)
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

    st.markdown('</div>', unsafe_allow_html=True)

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
    else:
        st.warning("⚠️ **No paper type selected.** Please select a paper type above to begin evaluation.")

    st.markdown("---")

    # Check if paper type selected and render section evaluator
    if not st.session_state.paper_type:
        st.warning("⚠️ **Please select a paper type above to begin evaluation.**")
    else:
        # Clear LLM history when switching tabs
        if st.session_state.active_tab != "Section Evaluator":
            if st.session_state.active_tab is not None:
                try:
                    cm.clear_history()
                except Exception:
                    pass
            st.session_state.active_tab = "Section Evaluator"

        # Initialize and render section evaluator (matches app.py exactly)
        if "section_eval_app" not in st.session_state:
            st.session_state.section_eval_app = SectionEvaluatorApp()

        st.session_state.section_eval_app.render_ui(
            files=st.session_state.get("file_data", {}),
            paper_type=st.session_state.paper_type
        )

# ============================================================================
# TAB 2: REFEREE REPORT (DEMO)
# ============================================================================
with tab2:
    st.header("Multi-Agent Referee Evaluation")

    # Update active tab
    if st.session_state.active_tab != "Referee Report":
        if st.session_state.active_tab is not None:
            try:
                cm.clear_history()
            except Exception:
                pass
        st.session_state.active_tab = "Referee Report"

    st.markdown("### 🎭 Demo: Multi-Agent Paper Evaluation")
    st.info("**Note:** This is a demonstration showing what the output would look like. Upload functionality coming soon!")

    # Agent Persona Descriptions
    st.markdown("### 👥 Meet the Review Panel")

    # Persona Selection Justification
    st.markdown("""
    <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #1976d2; margin: 20px 0;">
        <h4 style="margin-top: 0;">🎯 Why These Personas Were Chosen</h4>
        <p><strong>Weights:</strong> Empiricist (40%), Policymaker (35%), Historian (25%)</p>
        <p><strong>Justification:</strong> The paper's rigorous empirical analysis of textual data for economic forecasting and policy relevance,
        coupled with its historical contextualization of the Beige Book, makes the Empiricist, Policymaker, and Historian personas most crucial.</p>
        <ul style="margin: 10px 0;">
            <li><strong>Empiricist (40%):</strong> Primary focus on statistical validity and econometric rigor for sentiment analysis</li>
            <li><strong>Policymaker (35%):</strong> Assesses real-world applicability and policy implications of forecasting models</li>
            <li><strong>Historian (25%):</strong> Evaluates methodological advancement and literature contextualization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="persona-box empiricist-box">
            <h4>📊 Empiricist</h4>
            <p><strong>Focus:</strong> Empirical validity, statistical rigor, data quality</p>
            <p><strong>Criteria:</strong> Identification strategy, model specification, standard errors, robustness</p>
            <p><strong>Personality:</strong> Methodologically rigorous, critical of statistical claims</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="persona-box historian-box">
            <h4>📚 Historian</h4>
            <p><strong>Focus:</strong> Literature context, gap analysis, citation accuracy</p>
            <p><strong>Criteria:</strong> Lineage, novelty claims, historical grounding</p>
            <p><strong>Personality:</strong> Knowledgeable, despises false novelty claims</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="persona-box policymaker-box">
            <h4>🏛️ Policymaker</h4>
            <p><strong>Focus:</strong> Policy applicability, welfare implications, real-world impact</p>
            <p><strong>Criteria:</strong> Actionable insights, feasibility, practical relevance</p>
            <p><strong>Personality:</strong> Impact-oriented, values implementable recommendations</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Architecture Diagram
    st.markdown('<div class="architecture-box">', unsafe_allow_html=True)
    st.subheader("🤖 Multi-Agent Debate Architecture")

    st.markdown("**Debate Process:**")

    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
    <div style="margin-bottom: 25px;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 25px; border-radius: 5px; font-weight: bold; font-size: 18px; min-width: 50px;">
                1.
            </div>
            <div style="margin-left: 20px; color: #333;">
                <strong style="font-size: 18px;">Independent Evaluation</strong><br/>
                <span style="font-size: 14px;">Each agent evaluates independently and provides domain-specific verdict with evidence</span>
            </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
    <div style="margin-bottom: 25px;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 25px; border-radius: 5px; font-weight: bold; font-size: 18px; min-width: 50px;">
                2.
            </div>
            <div style="margin-left: 20px; color: #333;">
                <strong style="font-size: 18px;">Cross-Examination</strong><br/>
                <span style="font-size: 14px;">Agents read each other's reports, challenge assumptions, and engage in debate</span>
            </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #f8f9fa; padding: 15px; padding-left: 40px; border-radius: 8px; border-left: 4px solid #9fa8da; margin: 10px 0 10px 40px;">
        <div style="margin-bottom: 12px;">
            <span style="background: #9fa8da; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px;">2a</span>
            <strong style="margin-left: 10px; font-size: 15px;">Debate Round</strong><br/>
            <span style="margin-left: 10px; font-size: 13px; color: #555;">Agents challenge each other's reasoning and synthesize insights</span>
        </div>
        <div style="margin-bottom: 12px;">
            <span style="background: #9fa8da; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px;">2b</span>
            <strong style="margin-left: 10px; font-size: 15px;">Verdict Trajectory Analysis</strong><br/>
            <span style="margin-left: 10px; font-size: 13px; color: #555;">Visual diagram showing how verdicts evolved and which peers influenced changes</span>
        </div>
        <div>
            <span style="background: #9fa8da; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px;">2c</span>
            <strong style="margin-left: 10px; font-size: 15px;">Final Amendments</strong><br/>
            <span style="margin-left: 10px; font-size: 13px; color: #555;">Agents integrate peer feedback and submit final verdicts (PASS/REVISE/FAIL)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
    <div style="margin-bottom: 15px;">
        <div style="display: flex; align-items: center;">
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 25px; border-radius: 5px; font-weight: bold; font-size: 18px; min-width: 50px;">
                3.
            </div>
            <div style="margin-left: 20px; color: #333;">
                <strong style="font-size: 18px;">Editor Decision</strong><br/>
                <span style="font-size: 14px;">Senior Editor synthesizes verdicts and applies decision rules to produce final report</span>
            </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Key Insight:** Multi-agent debate prevents single-perspective bias. Agents update beliefs through constructive friction, leading to more robust and fair evaluation decisions.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ROUND 1: Independent Evaluation
    st.markdown('<div class="round-header">⚡ ROUND 1: INDEPENDENT EVALUATION</div>', unsafe_allow_html=True)

    # Empiricist Round 1
    with st.expander("📊 **EMPIRICIST** - Round 1 Assessment", expanded=False):
        st.markdown("""
        <div class="persona-box empiricist-box">

        <h4>1. Verdict</h4>
        <p class="verdict-revise">⚠️ REVISE</p>

        <h4>2. Empirical Audit</h4>
        <p>The paper investigates the predictive and nowcasting power of Beige Book sentiment using FinBERT for sentiment extraction
        and linear/logistic regression models. The use of FinBERT is sound, and text preprocessing is standard. The exclusion of
        COVID-19 period (1980:Q2 to 2019:Q4) is reasonable. Model specifications for GDP growth nowcasting/forecasting and recession
        prediction are appropriate for time series analysis. Panel regressions for regional analysis with bank and year fixed effects
        are well-specified.</p>

        <p><strong>⚠️ CRITICAL FLAW - STANDARD ERRORS:</strong></p>
        <p>A significant concern arises from the treatment of standard errors in all regression analyses. For time series regressions
        (Tables 1, 2, 4, 5) and panel regressions (Table 6), the paper only states "Standard errors appear in parentheses."</p>

        <p><strong>For time series data:</strong> Macroeconomic time series are highly likely to exhibit serial correlation. Standard
        OLS/MLE standard errors would be biased and inconsistent, leading to incorrect inference. Robust standard errors, such as
        Newey-West, are typically required.</p>

        <p><strong>For panel data:</strong> Errors are almost certainly correlated within each bank over time. Standard errors should
        be clustered at the bank level at a minimum. Without this, the reported significance levels are likely overstated.</p>

        <p>This omission of appropriate robust or clustered standard errors is a <strong>critical flaw</strong> that undermines the
        statistical validity of all quantitative results.</p>

        <h4>3. Source Evidence</h4>
        <ul>
            <li>"Standard errors appear in parentheses: *p< 0.1; **p< 0.05; ***p< 0.01." (Tables 1, 2, 4, 5, 6) - No mention of clustering or HAC</li>
            <li>"We conduct our empirical analysis from 1980:Q2 to 2019:Q4." (p. 7)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Historian Round 1
    with st.expander("📚 **HISTORIAN** - Round 1 Assessment", expanded=False):
        st.markdown("""
        <div class="persona-box historian-box">

        <h4>1. Verdict</h4>
        <p class="verdict-pass">✅ PASS</p>

        <h4>2. Lineage & Context</h4>
        <p>This paper builds upon a well-established lineage of economic literature on the Federal Reserve's Beige Book. Early
        predecessors include Balke and Petersen (2002) and Balke, Fulmer, and Zhang (2017), who pioneered converting Beige Book
        text into numerical scores. Further advancements include Sadique et al. (2013), who demonstrated predictive power for
        economic turning points, and Gascon and Werner (2022), who utilized keyword searches. Recent works like Burke and Nelson
        (2025) and Gascon and Martorana (2025) continue exploring the Beige Book's utility.</p>

        <h4>3. Gap Analysis</h4>
        <p>The paper claims to fill several gaps:</p>

        <p><strong>1. Methodological Advancement (FinBERT):</strong> Moving beyond dictionary-based approaches by employing
        FinBERT represents a genuine methodological advancement. ✅ <em>Gap is real and convincingly filled.</em></p>

        <p><strong>2. Comprehensive Control Variables:</strong> Explicitly controlling for yield spread, news sentiment, and
        SPF forecasts is a rigorous approach to isolating the unique contribution of Beige Book sentiment. ✅ <em>Gap is
        addressed with appropriate econometric analysis.</em></p>

        <p><strong>3. Regional Economic Activity:</strong> While Burke and Nelson (2025) use regional Beige Book data for
        national forecasts, this paper's specific focus on explaining <em>regional GDP growth</em> and identifying
        <em>inter-regional spillover effects</em> is indeed a distinct contribution. ✅ <em>Genuine extension addressed
        with panel regressions.</em></p>

        <p><strong>4. Richer Context:</strong> Using LDA for topic modeling offers valuable qualitative insights into how
        economic concerns evolved historically. ✅ <em>Convincingly demonstrated.</em></p>

        <h4>4. Source Evidence</h4>
        <p><em>"However, most of the literature is based on the entire body of text and uses a dictionary approach... We also
        show that the Beige Book can shed light on regional economic activity as well, which the literature has not touched upon."</em> (p. 5)</p>
        </div>
        """, unsafe_allow_html=True)

    # Policymaker Round 1
    with st.expander("🏛️ **POLICYMAKER** - Round 1 Assessment", expanded=False):
        st.markdown("""
        <div class="persona-box policymaker-box">

        <h4>1. Verdict</h4>
        <p class="verdict-pass">✅ PASS</p>

        <h4>2. Policy Applicability</h4>
        <p>This paper offers several actionable insights for central banks, government agencies, and research institutions:</p>

        <p><strong>1. Enhanced Real-Time Economic Assessment:</strong> Policymakers can integrate the FinBERT-derived Beige Book
        sentiment index into real-time economic monitoring dashboards. Its robust performance in nowcasting GDP growth and
        forecasting recessions makes it a valuable complementary tool.</p>

        <p><strong>2. Early Warning System for Recessions:</strong> The Beige Book sentiment's superior ability to predict economic
        recessions and identify business cycle turning points suggests it can serve as an early warning indicator for downside risks.</p>

        <p><strong>3. Granular Regional Policy Insights:</strong> Reserve Bank-level Beige Book sentiment explains regional GDP
        growth, with spillover effects between districts providing crucial intelligence for understanding heterogeneous economic
        conditions across Federal Reserve Districts.</p>

        <p><strong>4. Contextualizing Economic Shocks:</strong> Topical analysis (keyword searches and LDA) provides rich historical
        context for economic events, helping policymakers differentiate between unique economic episodes.</p>

        <h4>3. Welfare Implications</h4>
        <p>A deeper, more timely understanding of economic conditions can lead to more targeted and effective policy interventions:</p>
        <ul>
            <li><strong>More Timely Monetary Policy:</strong> Earlier recession signals enable more nimble policy decisions</li>
            <li><strong>Reduced Economic Volatility:</strong> Proactive responses can help stabilize economic cycles</li>
            <li><strong>Improved Resource Allocation:</strong> Regional insights allow for better-targeted federal resources</li>
            <li><strong>Enhanced Public Trust:</strong> More comprehensive understanding leads to better communication</li>
        </ul>

        <h4>4. Source Evidence</h4>
        <p><em>"Our most robust finding is the Beige Book's remarkable predictive power for economic recessions. The FinBERT
        sentiment index demonstrates statistically significant explanatory power in both nowcasting and forecasting recessions."</em> (p. 31)</p>
        <p><em>"A major contribution of our analysis is demonstrating that the Beige Book's informational value extends to regional
        economic activity... We find substantial spillover effects."</em> (p. 31)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ROUND 2A: Cross-Examination
    st.markdown('<div class="round-header">🔄 ROUND 2A: CROSS-EXAMINATION</div>', unsafe_allow_html=True)
    st.markdown("*Agents read each other's reports and engage in constructive debate...*")

    with st.expander("📊 **EMPIRICIST** - Cross-Examination", expanded=False):
        st.markdown("""
        <div class="persona-box empiricist-box">
        <h4>Empirical Audit (Updated)</h4>
        <p>The Policymaker's enthusiasm for the Beige Book sentiment as an "early warning system for recessions" and for "enhanced
        real-time economic assessment" is largely validated by the econometric analysis, particularly the strong nowcasting and
        recession forecasting results. The Historian's recognition of methodological advancement using FinBERT and rigorous inclusion
        of control variables aligns with sound econometric practice.</p>

        <p><strong>However, a critical econometric detail remains unaddressed:</strong> The paper states "Standard errors appear in
        parentheses" in all regression tables, but it does not specify whether these are robust to heteroskedasticity or clustered
        for potential autocorrelation in time series data or panel-specific effects. Given that the data are time series and panel
        data, uncorrected standard errors could lead to inflated t-statistics and spurious claims of statistical significance.</p>

        <h4>Constructive Pushback</h4>
        <p>While the Policymaker and Historian praise the "robust performance" and "statistically significant explanatory power,"
        this assessment is premature without proper validation of reported statistical significance.</p>

        <h4>Clarification Requests</h4>
        <ul>
            <li><strong>To Policymaker:</strong> Given that Beige Book sentiment shows "minimal evidence of contributing to
            out-of-sample forecasting performance" for GDP growth (Table 3), how does this affect confidence in using it as
            an early warning system?</li>
            <li><strong>To Historian:</strong> Could you elaborate on whether the type of standard errors used (HAC for time series,
            clustered for panel) was considered in assessing the convincingness of empirical results?</li>
        </ul>

        <h4>Updated Verdict</h4>
        <p class="verdict-revise">⚠️ REVISE</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📚 **HISTORIAN** - Cross-Examination", expanded=False):
        st.markdown("""
        <div class="persona-box historian-box">
        <h4>Lineage & Context (Updated)</h4>
        <p>This paper builds effectively on existing literature by introducing more advanced NLP techniques (FinBERT, LDA) and
        conducting a rigorous empirical analysis with comprehensive controls. The extension to specific regional economic outcomes
        and inter-regional spillovers constitutes a genuine and valuable contribution.</p>

        <p><strong>However, the convincingness of how these gaps are filled, particularly regarding quantitative claims of
        "statistically significant explanatory power," is severely undermined by the Empiricist's critical audit.</strong></p>

        <p>The claimed gaps are largely real and represent genuine advancements. Yet the pervasive issue with standard error
        calculations means that statistical significance reported in Tables 1, 2, 4, 5, and 6 cannot be trusted. While qualitative
        insights from topical analysis remain valuable, the core quantitative claims lack robust statistical foundation.</p>

        <h4>Cross-Domain Insights</h4>
        <p>The Empiricist's report highlights a critical econometric flaw in standard error calculation that fundamentally compromises
        statistical validity of all quantitative claims. From an Economic Historian's perspective, while the paper identifies
        legitimate gaps and proposes innovative ways to fill them, the evidence presented for quantitative aspects is currently
        unreliable.</p>

        <h4>Constructive Pushback</h4>
        <p>I must strongly push back against the Policymaker's "PASS" verdict and unreserved praise for "robust performance." The
        Empiricist clearly states that "the omission of appropriate method for calculating standard errors is a critical flaw"
        that "undermines statistical inference of all quantitative results." Claims of predictive power are only as strong as their
        statistical foundation.</p>

        <h4>Updated Verdict</h4>
        <p class="verdict-revise">⚠️ REVISE</p>
        <p><em>Genuine methodological advancement undermined by statistical validity concerns.</em></p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🏛️ **POLICYMAKER** - Cross-Examination", expanded=False):
        st.markdown("""
        <div class="persona-box policymaker-box">
        <h4>Policy Applicability (Updated)</h4>
        <p>The paper's core premise—leveraging qualitative insights from the Beige Book using advanced NLP for real-time economic
        assessment—holds significant promise. The ability to nowcast GDP growth and forecast recessions with regional granularity
        offers a complementary data source. The identification of regional spillover effects is particularly valuable.</p>

        <p><strong>However, the Empiricist identifies a critical flaw:</strong> The reported statistical significance cannot be
        trusted due to the likely omission of appropriate robust or clustered standard errors. If standard errors are understated,
        then claims of "meaningful explanatory power" and "superior performance" are undermined.</p>

        <h4>Cross-Domain Insights</h4>
        <p>My initial assessment of policy relevance is largely validated by both peer reports regarding innovative NLP application
        and novel regional analysis. The Empiricist acknowledges the "rich and novel dataset" and "sound choice" of FinBERT, while
        the Historian praises genuine methodological advancement.</p>

        <p><strong>However, while the Historian gives a "PASS" initially, this claim is directly contradicted by the Empiricist's
        critical finding about standard errors.</strong> The methodological soundness for quantitative claims is not as "appropriate"
        as suggested.</p>

        <h4>Clarification Request</h4>
        <ul>
            <li><strong>To Empiricist:</strong> Could you provide a concrete example where a currently significant coefficient is
            susceptible to becoming insignificant if proper standard errors were applied, and explain the policy implication?</li>
            <li><strong>To Historian:</strong> Given the Empiricist's finding of critical flaw in statistical validity, how do you
            reconcile your assessment that the paper "fills gaps convincingly"?</li>
        </ul>

        <h4>Updated Verdict</h4>
        <p class="verdict-revise">⚠️ REVISE</p>
        <p><em>Strong conceptual contribution, but empirical execution insufficient.</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ROUND 2B: Verdict Trajectory Analysis
    st.markdown('<div class="round-header">📊 ROUND 2B: VERDICT TRAJECTORY ANALYSIS</div>', unsafe_allow_html=True)

    # Create the trajectory table
    st.markdown("""
    <style>
    .trajectory-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-radius: 10px;
        overflow: hidden;
    }
    .trajectory-table thead {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .trajectory-table th {
        padding: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
    }
    .trajectory-table td {
        padding: 12px;
        text-align: center;
        border-bottom: 1px solid #e0e0e0;
    }
    .trajectory-table tbody tr:hover {
        background: #f5f5f5;
    }
    .verdict-pass-icon { color: #4caf50; font-size: 20px; }
    .verdict-fail-icon { color: #f44336; font-size: 20px; }
    .verdict-revise-icon { color: #ff9800; font-size: 20px; }
    .status-mixed { background: #fff9c4; color: #f57f17; font-weight: bold; padding: 4px 8px; border-radius: 4px; }
    .status-converging { background: #ffe0b2; color: #e65100; font-weight: bold; padding: 4px 8px; border-radius: 4px; }
    .status-rejected { background: #ffcdd2; color: #c62828; font-weight: bold; padding: 4px 8px; border-radius: 4px; }
    </style>

    <table class="trajectory-table">
        <thead>
            <tr>
                <th>Round</th>
                <th>📊 Empiricist</th>
                <th>📚 Historian</th>
                <th>🏛️ Policymaker</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="font-weight: bold;">Round 1: Independent</td>
                <td><span class="verdict-revise-icon">⚠️</span> REVISE</td>
                <td><span class="verdict-pass-icon">✅</span> PASS</td>
                <td><span class="verdict-pass-icon">✅</span> PASS</td>
                <td><span class="status-mixed">Mixed</span></td>
            </tr>
            <tr>
                <td style="font-weight: bold;">Round 2A: Cross-Exam</td>
                <td><span class="verdict-revise-icon">⚠️</span> REVISE</td>
                <td><span class="verdict-revise-icon">⚠️</span> REVISE</td>
                <td><span class="verdict-revise-icon">⚠️</span> REVISE</td>
                <td><span class="status-converging">Converging</span></td>
            </tr>
            <tr>
                <td style="font-weight: bold;">Round 2C: Final</td>
                <td><span class="verdict-revise-icon">⚠️</span> REVISE</td>
                <td><span class="verdict-revise-icon">⚠️</span> REVISE</td>
                <td><span class="verdict-revise-icon">⚠️</span> REVISE</td>
                <td><span class="status-rejected">Rejected</span></td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    # ROUND 2C: Final Amendments
    st.markdown('<div class="round-header">⚖️ ROUND 2C: FINAL AMENDMENTS</div>', unsafe_allow_html=True)
    st.markdown("*Agents submit their final verdicts after integrating peer feedback...*")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="persona-box empiricist-box">
        <h4>📊 Empiricist</h4>
        <h5>Insights Absorbed</h5>
        <p>The cross-examination revealed that while the Policymaker and Historian recognize the paper's conceptual value and
        methodological advancements, they also acknowledge the critical econometric flaw regarding unspecified standard errors.
        This issue directly impacts statistical validity of all quantitative findings.</p>

        <h5>Final Assessment</h5>
        <p>The paper presents a methodologically sound approach using advanced NLP techniques and addresses important gaps in
        the literature. However, the critical flaw of unspecified standard errors in all regression analyses fundamentally
        undermines statistical validity of quantitative findings.</p>

        <p class="verdict-revise"><strong>Final Verdict: ⚠️ REVISE</strong></p>
        <p><em>Methodologically sound with important gaps addressed, but requires revision for robust econometric inference.</em></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="persona-box historian-box">
        <h4>📚 Historian</h4>
        <h5>Insights Absorbed</h5>
        <p>The debate significantly refined my understanding. The Empiricist's rigorous critique regarding unspecified standard
        errors forced me to re-evaluate the "convincingness" of how quantitative gaps were filled. While the methodology is
        sound, the evidence presented to support statistical significance is currently unreliable.</p>

        <h5>Final Assessment</h5>
        <p>The paper proposes genuine advancements in applying NLP to the Beige Book (FinBERT, LDA) and offers novel regional
        analysis, addressing real gaps in the literature. However, the critical flaw concerning statistical validity of standard
        errors fundamentally undermines the "convincingness" of these findings.</p>

        <p class="verdict-revise"><strong>Final Verdict: ⚠️ REVISE</strong></p>
        <p><em>Genuine advancements and novel analysis, but requires statistical rigor before full acceptance.</em></p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="persona-box policymaker-box">
        <h4>🏛️ Policymaker</h4>
        <h5>Insights Absorbed</h5>
        <p>The Empiricist's finding regarding standard errors is a critical flaw that directly impacts statistical validity of
        quantitative claims. My initial assessment of "robust performance" was premature without proper validation of reported
        statistical significance.</p>

        <h5>Final Assessment</h5>
        <p>While the methodological approach (FinBERT, LDA, regional analysis) is innovative and the type of information extracted
        is inherently valuable, the quantitative evidence supporting its predictive power is currently compromised. However, the
        paper's overall policy relevance and potential remains significant.</p>

        <p class="verdict-revise"><strong>Final Verdict: ⚠️ REVISE</strong></p>
        <p><em>Strong policy relevance and welfare implications, but statistical rigor required for quantitative claims.</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ROUND 3: Editor Decision
    st.markdown('<div class="round-header">📜 ROUND 3: EDITOR DECISION</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="persona-box editor-box">
    <h3>🏛️ Senior Editor's Final Decision</h3>

    <h4>Weight Calculation</h4>
    <ul style="list-style: none; padding-left: 0;">
        <li>📊 <strong>Empiricist</strong>: Verdict = REVISE (0.5) × Weight = 0.4 = <strong>0.20</strong></li>
        <li>🏛️ <strong>Policymaker</strong>: Verdict = REVISE (0.5) × Weight = 0.35 = <strong>0.175</strong></li>
        <li>📚 <strong>Historian</strong>: Verdict = REVISE (0.5) × Weight = 0.25 = <strong>0.125</strong></li>
    </ul>
    <p><strong>Total Weighted Score:</strong> 0.20 + 0.175 + 0.125 = <strong>0.50</strong></p>

    <h4>Debate Synthesis</h4>
    <p>The panel unanimously agrees that while the paper presents methodologically sound advancements in NLP application and
    valuable regional analysis, a critical econometric flaw regarding unspecified standard errors fundamentally undermines
    the statistical validity of its quantitative findings. This necessitates a revision to ensure robust inference before
    the paper's claims of predictive power and policy relevance can be fully accepted.</p>

    <h3 style="color: #c62828; margin-top: 30px;">📋 FINAL DECISION: ❌ REJECT AND RESUBMIT</h3>

    <hr>

    <h4>Official Referee Report</h4>

    <p>Dear Authors,</p>

    <p>We have reviewed your manuscript and appreciate the innovative application of NLP techniques, particularly FinBERT and LDA,
    to the Beige Book for regional economic analysis. The panel recognizes the genuine advancements and the filling of important
    literature gaps, as noted by all reviewers.</p>

    <p>However, a critical econometric flaw has been identified that fundamentally undermines the statistical validity of your
    quantitative findings. The Empiricist highlighted the "unspecified nature of standard errors in the regression analyses,"
    stating this "directly impacts the statistical validity of all quantitative findings, potentially rendering claims of
    'statistically significant explanatory power' unreliable." The reviewer provided a concrete example showing "how a significant
    coefficient could become insignificant under clustered standard errors," underscoring the gravity of this issue for your
    regional analysis.</p>

    <p>The Policymaker concurs, noting that this flaw "necessitates a revision of the paper's quantitative claims" and has
    "tempered [their] initial enthusiasm for the paper's 'robust performance' and 'statistically significant explanatory power'."
    The Historian also agrees, stating that while the methodology is sound, "the evidence presented to support the statistical
    significance of the findings is currently unreliable."</p>

    <ol>
        <li><strong>Address the unspecified standard errors in all regression analyses</strong> to ensure robust econometric inference.</li>
        <li>Specify whether HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors are used for time series data.</li>
        <li>Specify whether clustered standard errors are used for panel data regressions.</li>
        <li>Re-evaluate all reported statistical significance with properly specified standard errors.</li>
    </ol>

    <p style="margin-top: 20px;">We look forward to a revised submission that rectifies these critical statistical issues. Your
    work addresses an important research question, and with proper econometric rigor, it has the potential to make a significant
    contribution to the field.</p>

    <p>Sincerely,<br/>
    The Editorial Team</p>

    <p>We wish you success in future endeavors.</p>

    <p>Sincerely,<br>
    The Senior Editor</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # System Notes
    with st.expander("💡 Demo Notes & Observations", expanded=False):
        st.markdown("""
        ### 🔍 Observations from this Demo

        **Strengths of Multi-Agent Debate:**
        - ✅ Constructive discussion encouraged belief updates
        - ✅ Historian shifted verdict after Empiricist's evidence (PASS → FAIL)
        - ✅ Cross-domain synthesis prevented single-perspective bias
        - ✅ No significant hallucinations detected in debate

        **Points for Consideration:**
        - ⚠️ Empiricist may be overly punitive on negative adjusted R² issue
        - ⚠️ Title misspecification noted in original paper
        - ⚠️ Personas converge strongly around mathematical evidence

        **System Design:**
        - Three specialized agents with distinct evaluation criteria
        - Multi-round debate with cross-examination and trajectory analysis
        - Hierarchical decision rules (Technical Kill Switch)
        - Evidence-based argumentation required

        **Future Enhancements:**
        - Real-time paper upload and evaluation
        - Customizable persona weights
        - Interactive debate visualization
        - Comparative analysis across multiple papers
        """)

# Footer
st.markdown("---")
st.caption("**Research Evaluation Agent Demo** | Federal Reserve System | Powered by Claude & Multi-Agent Debate")
