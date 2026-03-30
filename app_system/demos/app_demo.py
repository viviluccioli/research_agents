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
st.set_page_config(layout="wide", page_title="Evaluation Agent Demo", page_icon="📊")

# Custom CSS for styling
st.markdown("""
<style>
    .persona-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .mathematician-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
    }
    .historian-box {
        background-color: #f3e5f5;
        border-left: 5px solid #7b1fa2;
    }
    .visionary-box {
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
st.title("📊 Research Evaluation Agent")
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

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="persona-box mathematician-box">
            <h4>🔢 Mathematician</h4>
            <p><strong>Focus:</strong> Technical rigor, empirical validity, mathematical accuracy</p>
            <p><strong>Criteria:</strong> Model specification, statistical inference, derivation correctness</p>
            <p><strong>Personality:</strong> Harsh, detail-oriented, doesn't care about "story"</p>
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
        <div class="persona-box visionary-box">
            <h4>🚀 Visionary</h4>
            <p><strong>Focus:</strong> Innovation, paradigm-shifting potential, creativity</p>
            <p><strong>Criteria:</strong> Novelty score, field impact, intellectual risk</p>
            <p><strong>Personality:</strong> Bold, values breakthroughs over incrementalism</p>
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

    # Mathematician Round 1
    with st.expander("🔢 **MATHEMATICIAN** - Round 1 Assessment", expanded=False):
        st.markdown("""
        <div class="persona-box mathematician-box">

        <h4>1. Verdict</h4>
        <p class="verdict-fail">❌ FAIL</p>

        <h4>2. Rationale</h4>
        <p>The paper employs a mixed methodology, combining natural language processing (NLP) techniques with econometric modeling
        to analyze the Beige Book. While the application of FinBERT for sentiment scoring and LDA for topic modeling is methodologically
        sound, the econometric implementation suffers from fundamental flaws.</p>

        <p><strong>⚠️ CRITICAL FLAW IDENTIFIED:</strong></p>
        <p>In Table 6, Model (1), which includes both bank and year fixed effects, the reported <strong>Adjusted R² is -0.111</strong>.
        This value indicates that the model explains less variance than a simple model with only an intercept. A negative Adjusted R²
        implies a very poor fit and suggests that the included predictors do not collectively improve upon a basic benchmark.</p>

        <p>This directly <strong>contradicts the authors' claim</strong> that "The R Squares are relatively high, indicating that a lot
        of the variation in economic growth rates across Reserve Banks and through time can be captured by our panel model" (Page 22).</p>

        <p>This fundamental misrepresentation of model performance undermines empirical validity and constitutes a critical breach of scientific reporting standards.</p>

        <h4>3. Evidence</h4>
        <ul>
            <li>"Table 6: Federal Reserve Bank-Level Panel Regressions" (Page 22) → Model (1) Adjusted R²: <strong>-0.111</strong></li>
            <li>Authors' claim: "The R Squares are relatively high, indicating that a lot of the variation in economic growth rates across Reserve Banks and through time can be captured by our panel model" (Page 22)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Historian Round 1
    with st.expander("📚 **HISTORIAN** - Round 1 Assessment", expanded=False):
        st.markdown("""
        <div class="persona-box historian-box">

        <h4>1. Verdict</h4>
        <p class="verdict-pass">✅ PASS</p>

        <h4>2. Rationale</h4>
        <p>This paper is <strong>firmly grounded</strong> in the existing literature on using the Federal Reserve's Beige Book for
        economic analysis, while also integrating advancements in natural language processing (NLP).</p>

        <p>The authors explicitly acknowledge foundational work (Balke and Petersen 2002; Balke, Fulmer and Zhang 2017;
        Gascon and Werner 2022; Sadique et al. 2013) and effectively identify <strong>several genuine gaps</strong>:</p>

        <p><strong>1. Methodological Advancement:</strong> Leverages "new machine learning models" like FinBERT for sentiment
        analysis and LDA for topic modeling, offering more sophisticated extraction than previous dictionary approaches.</p>

        <p><strong>2. Regional Granularity:</strong> Authors explicitly state that "the literature has not touched upon" the Beige
        Book's ability to shed light on regional economic activity. Their analysis of Reserve Bank-level sentiment and spillover
        effects directly fills this void.</p>

        <p><strong>3. Richer Historical Context:</strong> Provides context on "how different topics evolved throughout the past
        half century," including international developments and distinct characteristics of different recessionary periods.</p>

        <p>The paper demonstrates strong contextualization and genuine gap identification, meeting standards for literature grounding.</p>

        <h4>3. Evidence</h4>
        <p><em>"However, most of the literature is based on the entire body of text and uses a dictionary approach for extracting
        information from the Beige Books. In this paper, we provide a general overview of how sentiment measured by new machine
        learning models can help nowcast and predict economic activity and recessions... We also show that the Beige Book can shed
        light on regional economic activity as well, which the literature has not touched upon."</em> (Page 5)</p>
        </div>
        """, unsafe_allow_html=True)

    # Visionary Round 1
    with st.expander("🚀 **VISIONARY** - Round 1 Assessment", expanded=False):
        st.markdown("""
        <div class="persona-box visionary-box">

        <h4>1. Verdict</h4>
        <p class="verdict-pass">✅ PASS</p>

        <h4>2. Rationale</h4>
        <p>This paper makes a <strong>compelling case for a re-evaluation of "soft information"</strong> in economic analysis,
        particularly within central banking.</p>

        <p>By systematically demonstrating the unique predictive power of the Beige Book's qualitative insights—especially concerning
        regional economic interdependencies and recession forecasting—it <strong>challenges the prevailing paradigm</strong> that
        often prioritizes quantitative, hard data.</p>

        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>Beige Book sentiment significantly explains regional GDP growth variation</li>
            <li>Substantial spillover effects: each district's activity correlates with sentiment about other districts</li>
            <li>Remarkable predictive power for economic recessions, even controlling for canonical predictors (yield curve)</li>
        </ul>

        <p>If these findings hold, it could lead to a <strong>significant shift</strong> in how policymakers gather, interpret, and
        integrate anecdotal intelligence, moving it from a supplementary input to a core component of real-time economic assessment.
        The novel approach demonstrates significant paradigm-shifting potential.</p>

        <h4>3. Evidence</h4>
        <p><em>"A major contribution of our analysis is demonstrating that the Beige Book's informational value extends to regional
        economic activity... Even more intriguingly, we find substantial spillover effects: each district's economic activity is
        strongly correlated with sentiment expressed about other districts beyond its own sentiment. This finding has important
        implications for understanding how regional economic conditions propagate through the national economy."</em> (Page 31)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ROUND 2A: Cross-Examination
    st.markdown('<div class="round-header">🔄 ROUND 2A: CROSS-EXAMINATION</div>', unsafe_allow_html=True)
    st.markdown("*Agents read each other's reports and engage in constructive debate...*")

    with st.expander("🔢 **MATHEMATICIAN** - Cross-Examination", expanded=False):
        st.markdown("""
        <div class="persona-box mathematician-box">
        <h4>Synthesis with Other Agents</h4>
        <p><strong>With Historian:</strong> I acknowledge the Historian's robust assessment of the paper's grounding in existing literature and clear identification
        of genuine gaps. This provides solid motivation for the research questions. However, <strong>methodological soundness and
        empirical validity are paramount</strong>. If the econometric implementation suffers from fundamental issues such as poor
        model fit in key specifications, the claimed advancements become tenuous.</p>

        <p><strong>With Visionary:</strong> I appreciate the identification of "unique predictive power" and "inter-district spillover effects." These are significant
        claims that could have profound implications. However, the validity hinges entirely on empirical rigor. <strong>If the
        statistical foundations are weak</strong>—due to issues like uncorrected standard errors or models that fail to fit the
        data—then the "paradigm potential" is severely undermined.</p>

        <h4>Constructive Pushback</h4>
        <p>The Visionary's enthusiasm for "remarkable predictive power" and "substantial spillover effects" is <strong>premature
        given the econometric issues</strong>. Specifically:</p>
        <ul>
            <li>No explicit statement whether standard errors are robust to heteroskedasticity and autocorrelation</li>
            <li>Table 6, Model (1): Adjusted R² of -0.111 contradicts any claim of "significant explanatory power"</li>
            <li>This fundamentally weakens evidence for regional effects</li>
        </ul>

        <h4>Questions for Authors</h4>
        <ol>
            <li>Could the authors confirm whether standard errors in Tables 1-6 are robust? If not, provide updated analysis.</li>
            <li>Regarding Table 6, Model (1): Please explain the negative Adjusted R² and how this impacts interpretation.</li>
        </ol>

        <h4>Updated Verdict</h4>
        <p class="verdict-fail">❌ FAIL</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📚 **HISTORIAN** - Cross-Examination", expanded=False):
        st.markdown("""
        <div class="persona-box historian-box">
        <h4>Synthesis with Other Agents</h4>
        <p><strong>With Mathematician:</strong> I accept the Mathematician's rigorous audit, particularly the finding of a negative Adjusted R² for Model (1) in Table 6.
        This is not merely a technical detail but a <strong>fundamental flaw that directly contradicts the authors' claims</strong>
        regarding the explanatory power of their regional panel model.</p>

        <p>The authors state, "The R Squares are relatively high, indicating that a lot of the variation in economic growth rates
        across Reserve Banks and through time can be captured by our panel model" (Page 22), which is a <strong>clear misrepresentation</strong>
        of a negative R².</p>

        <p><strong>Impact on My Domain:</strong> From my historical perspective, this <strong>invalidates a core claim of novelty</strong>: that the Beige Book sheds
        light on regional economic activity "which the literature has not touched upon" (Page 5). If the evidence for this novel
        regional insight is flawed and misrepresented, the entire argument for this aspect of the paper's contribution collapses.</p>

        <h4>Constructive Pushback</h4>
        <p>The Visionary's praise for "discovery of inter-district spillover effects" and the paper's ability to demonstrate regional
        insights is directly undermined by the Mathematician's findings. The regional analysis was presented as a significant, novel
        contribution, and if the underlying statistical evidence is flawed, the scope of "paradigm potential" for regional insights
        is severely diminished.</p>

        <h4>Questions for Authors</h4>
        <ol>
            <li><strong>To Mathematician:</strong> Does the flaw in Model (1) cast doubt on Model (2) or other regional claims?</li>
            <li><strong>To Visionary:</strong> Given the negative Adjusted R² and misrepresentation, how do you reconcile this with
            your assessment of "innovation" and "paradigm potential" for regional analysis?</li>
        </ol>

        <h4>Updated Verdict</h4>
        <p class="verdict-fail">❌ FAIL</p>
        <p><em>Misrepresentation of key finding undermines scientific integrity.</em></p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🚀 **VISIONARY** - Cross-Examination", expanded=False):
        st.markdown("""
        <div class="persona-box visionary-box">
        <h4>Synthesis with Other Agents</h4>
        <p><strong>With Mathematician:</strong> I accept the Mathematician's finding regarding the severe empirical flaw in Table 6, Model (1), where the reported
        Adjusted R² of -0.111 directly contradicts the authors' claim of "relatively high R Squares." This significantly undermines
        confidence in the robustness of the regional analysis's empirical demonstration.</p>

        <p>While the <strong>idea</strong> of extracting regional "soft information" remains conceptually novel and impactful, the
        presented evidence for this particular specification is demonstrably weak and the authors' interpretation is misleading.</p>

        <p><strong>With Historian:</strong> I fully embrace the Historian's assessment that this paper addresses genuine gaps by leveraging "new machine learning models"
        for a more sophisticated analysis of the Beige Book. The systematic extraction of soft information at a regional level
        represents a significant conceptual advancement.</p>

        <h4>Constructive Pushback</h4>
        <p>While the Historian identifies "regional granularity" as a key gap filled by this paper, the Mathematician's audit reveals
        a critical weakness in the empirical evidence. The discrepancy means that while the <strong>idea</strong> of filling this gap
        is novel and conceptually powerful, the <strong>demonstration</strong> of its success is empirically unsound.</p>

        <h4>Questions for Authors</h4>
        <ol>
            <li><strong>To Mathematician:</strong> Given that Model (2) shows positive Adjusted R² (0.232), does the flaw in Model (1)
            entirely invalidate the conceptual claim of regional insights, or only the specific demonstration in Model (1)?</li>
            <li><strong>To Historian:</strong> How does the negative Adjusted R² affect your assessment of the paper's success in
            "directly filling this void" regarding regional granularity?</li>
        </ol>

        <h4>Updated Verdict</h4>
        <p class="verdict-pass">✅ PASS (with strong reservations)</p>
        <p><em>Conceptual novelty remains, though empirical demonstration is flawed.</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ROUND 2B: Verdict Trajectory Analysis
    st.markdown('<div class="round-header">📊 ROUND 2B: VERDICT TRAJECTORY ANALYSIS</div>', unsafe_allow_html=True)

    # Create trajectory diagram using columns for better rendering (3 agents only)
    col_m, col_h, col_v = st.columns(3)

    with col_m:
        st.markdown("""
        <div style="text-align: center;">
            <div style="background: #e3f2fd; border: 3px solid #1976d2; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                <strong style="color: #1976d2;">🔢 Mathematician</strong><br/>
                <span style="font-size: 24px;">❌</span><br/>
                <span style="font-size: 12px; color: #666;">Round 1: FAIL</span>
            </div>
            <div style="font-size: 40px; color: #666; text-align: center;">↓</div>
            <div style="background: #ffebee; padding: 8px; border-radius: 5px; font-size: 11px; font-weight: bold; color: #c62828; margin: 10px 0;">
                NO CHANGE
            </div>
            <div style="font-size: 40px; color: #666; text-align: center;">↓</div>
            <div style="background: #e3f2fd; border: 3px solid #1976d2; border-radius: 10px; padding: 15px; margin-top: 10px;">
                <span style="font-size: 24px;">❌</span><br/>
                <span style="font-size: 12px; color: #666;">Round 2a: FAIL</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_h:
        st.markdown("""
        <div style="text-align: center;">
            <div style="background: #f3e5f5; border: 3px solid #7b1fa2; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                <strong style="color: #7b1fa2;">📚 Historian</strong><br/>
                <span style="font-size: 24px;">✅</span><br/>
                <span style="font-size: 12px; color: #666;">Round 1: PASS</span>
            </div>
            <div style="font-size: 40px; color: #666; text-align: center;">↓</div>
            <div style="background: #e3f2fd; padding: 8px; border-radius: 5px; font-size: 11px; font-weight: bold; color: #1976d2; margin: 10px 0;">
                INFLUENCED BY<br/>MATHEMATICIAN
            </div>
            <div style="font-size: 40px; color: #666; text-align: center;">↓</div>
            <div style="background: #f3e5f5; border: 3px solid #7b1fa2; border-radius: 10px; padding: 15px; margin-top: 10px;">
                <span style="font-size: 24px;">❌</span><br/>
                <span style="font-size: 12px; color: #666;">Round 2a: FAIL</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_v:
        st.markdown("""
        <div style="text-align: center;">
            <div style="background: #fff3e0; border: 3px solid #f57c00; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                <strong style="color: #f57c00;">🚀 Visionary</strong><br/>
                <span style="font-size: 24px;">✅</span><br/>
                <span style="font-size: 12px; color: #666;">Round 1: PASS</span>
            </div>
            <div style="font-size: 40px; color: #666; text-align: center;">↓</div>
            <div style="background: #ffebee; padding: 8px; border-radius: 5px; font-size: 11px; font-weight: bold; color: #c62828; margin: 10px 0;">
                NO CHANGE
            </div>
            <div style="font-size: 40px; color: #666; text-align: center;">↓</div>
            <div style="background: #fff3e0; border: 3px solid #f57c00; border-radius: 10px; padding: 15px; margin-top: 10px;">
                <span style="font-size: 24px;">✅</span><br/>
                <span style="font-size: 12px; color: #666;">Round 2a: PASS</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # st.markdown("---")
    # st.markdown("**Option 2: Verdict-Grouped Flow View**")

    # # Create simpler three-column layout
    # col_left, col_middle, col_right = st.columns([5, 3, 5])

    # with col_left:
    #     st.markdown("#### ⚡ ROUND 1")
    #     st.markdown("""
    #     <div style="background: #e8f5e9; border: 2px solid #4caf50; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
    #         <div style="text-align: center; font-weight: bold; color: #2e7d32; font-size: 16px; margin-bottom: 10px;">✅ PASS</div>
    #         <div style="background: #f3e5f5; border: 2px solid #7b1fa2; border-radius: 6px; padding: 8px; margin: 5px 0;">
    #             <span style="color: #7b1fa2; font-weight: bold;">📚 Historian →</span>
    #         </div>
    #         <div style="background: #fff3e0; border: 2px solid #f57c00; border-radius: 6px; padding: 8px; margin: 5px 0;">
    #             <span style="color: #f57c00; font-weight: bold;">🚀 Visionary →</span>
    #         </div>
    #     </div>
    #     <div style="background: #ffebee; border: 2px solid #f44336; border-radius: 8px; padding: 15px;">
    #         <div style="text-align: center; font-weight: bold; color: #c62828; font-size: 16px; margin-bottom: 10px;">❌ FAIL</div>
    #         <div style="background: #e3f2fd; border: 2px solid #1976d2; border-radius: 6px; padding: 8px; margin: 5px 0;">
    #             <span style="color: #1976d2; font-weight: bold;">🔢 Mathematician →</span>
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)

    # with col_middle:
    #     st.markdown("<br><br><br>", unsafe_allow_html=True)
    #     st.markdown("""
    #     <div style="text-align: center; margin-top: 20px;">
    #         <div style="margin: 25px 0;">
    #             <div style="font-size: 32px; color: #7b1fa2;">━━━➤</div>
    #             <div style="color: #7b1fa2; font-weight: bold; font-size: 10px; line-height: 1.2;">Influenced by<br/>MATHEMATICIAN</div>
    #         </div>
    #         <div style="margin: 35px 0;">
    #             <div style="font-size: 32px; color: #f57c00;">━━━➤</div>
    #             <div style="color: #f57c00; font-weight: bold; font-size: 10px;">No change</div>
    #         </div>
    #         <div style="margin: 80px 0 0 0;">
    #             <div style="font-size: 32px; color: #1976d2;">━━━➤</div>
    #             <div style="color: #1976d2; font-weight: bold; font-size: 10px;">No change</div>
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)

    # with col_right:
    #     st.markdown("#### 🔄 ROUND 2A")
    #     st.markdown("""
    #     <div style="background: #e8f5e9; border: 2px solid #4caf50; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
    #         <div style="text-align: center; font-weight: bold; color: #2e7d32; font-size: 16px; margin-bottom: 10px;">✅ PASS</div>
    #         <div style="background: #fff3e0; border: 2px solid #f57c00; border-radius: 6px; padding: 8px; margin: 5px 0;">
    #             <span style="color: #f57c00; font-weight: bold;">→ 🚀 Visionary</span>
    #         </div>
    #     </div>
    #     <div style="background: #ffebee; border: 2px solid #f44336; border-radius: 8px; padding: 15px;">
    #         <div style="text-align: center; font-weight: bold; color: #c62828; font-size: 16px; margin-bottom: 10px;">❌ FAIL</div>
    #         <div style="background: #e3f2fd; border: 2px solid #1976d2; border-radius: 6px; padding: 8px; margin: 5px 0;">
    #             <span style="color: #1976d2; font-weight: bold;">→ 🔢 Mathematician</span>
    #         </div>
    #         <div style="background: #f3e5f5; border: 2px solid #7b1fa2; border-radius: 6px; padding: 8px; margin: 5px 0;">
    #             <span style="color: #7b1fa2; font-weight: bold;">→ 📚 Historian</span>
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)

    # st.markdown("---")


    st.markdown("---")

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
                <th>🔢 Mathematician</th>
                <th>📚 Historian</th>
                <th>🚀 Visionary</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="font-weight: bold;">Round 1: Independent</td>
                <td><span class="verdict-fail-icon">❌</span> FAIL</td>
                <td><span class="verdict-pass-icon">✅</span> PASS</td>
                <td><span class="verdict-pass-icon">✅</span> PASS</td>
                <td><span class="status-mixed">Mixed</span></td>
            </tr>
            <tr>
                <td style="font-weight: bold;">Round 2A: Cross-Exam</td>
                <td><span class="verdict-fail-icon">❌</span> FAIL</td>
                <td><span class="verdict-fail-icon">❌</span> FAIL</td>
                <td><span class="verdict-pass-icon">✅</span> PASS</td>
                <td><span class="status-converging">Converging</span></td>
            </tr>
            <tr>
                <td style="font-weight: bold;">Round 2B: Final</td>
                <td><span class="verdict-fail-icon">❌</span> FAIL</td>
                <td><span class="verdict-fail-icon">❌</span> FAIL</td>
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
        <div class="persona-box mathematician-box">
        <h4>🔢 Mathematician</h4>
        <h5>Insights Absorbed</h5>
        <p>I acknowledge the conceptual value and relevance of the research questions highlighted by the Historian and Visionary.</p>

        <h5>Remaining Friction</h5>
        <ul>
            <li>Lack of robust standard errors specification</li>
            <li>No formal stationarity tests</li>
            <li>Severe misrepresentation of Adjusted R²</li>
        </ul>

        <p class="verdict-fail"><strong>Final Verdict: ❌ FAIL</strong></p>
        <p><em>Pervasive econometric issues prevent meeting basic validity standards. Unsalvageable without complete re-estimation.</em></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="persona-box historian-box">
        <h4>📚 Historian</h4>
        <h5>Insights Absorbed</h5>
        <p>The Mathematician's cross-examination significantly altered my initial positive assessment. The misrepresentation of negative R² reveals critical flaw in scientific integrity.</p>

        <h5>Remaining Friction</h5>
        <p>The authors' claim to fill the regional analysis gap is directly contradicted by presented evidence and its misinterpretation.</p>

        <p class="verdict-fail"><strong>Final Verdict: ❌ FAIL</strong></p>
        <p><em>Egregious misrepresentation of key finding constitutes critical breach of scientific integrity.</em></p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="persona-box visionary-box">
        <h4>🚀 Visionary</h4>
        <h5>Insights Absorbed</h5>
        <p>The Mathematician's findings significantly weaken the empirical basis for regional claims. Statistical significance of all coefficients is now questionable.</p>

        <h5>Remaining Friction</h5>
        <p>The core <em>conceptual</em> contribution—that "anecdotes matter" and systematically applying ML to qualitative documents yields unique insights—still embodies novelty. The issues are with the <strong>proof, not the premise</strong>.</p>

        <p class="verdict-revise"><strong>Final Verdict: ⚠️ REVISE</strong></p>
        <p><em>Genuinely novel idea, but empirical execution insufficient. Requires methodological fixes to substantiate claims.</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ROUND 3: Editor Decision
    st.markdown('<div class="round-header">📜 ROUND 3: EDITOR DECISION</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="persona-box editor-box">
    <h3>🏛️ Senior Editor's Final Decision</h3>

    <h4>Debate Synthesis</h4>
    <p>The panel converged significantly around the Mathematician's rigorous audit of the paper's empirical validity. The Historian,
    initially positive, updated their assessment to a "FAIL," explicitly acknowledging that the Mathematician's findings invalidated
    a core claim of novelty and constituted a critical breach of scientific integrity.</p>

    <p>While the Visionary maintained a "REVISE" verdict, acknowledging the paper's strong conceptual contribution, they too conceded
    that the empirical execution was insufficient to substantiate the claims, noting the issues were with the "proof, not the premise."</p>

    <h4>Final Decision Process</h4>
    <p><strong>Rule 1: Technical Kill Switch</strong> is directly applied here.</p>
    <ul>
        <li>✅ Mathematician marked "FAIL" due to "fatal empirical/math errors"</li>
        <li>✅ Historian marked "FAIL" citing "critical breach of scientific integrity"</li>
        <li>Since both marked FAIL for fundamental issues → <strong>PAPER IS REJECTED</strong></li>
    </ul>

    <h3 style="color: #c62828; margin-top: 30px;">📋 FINAL DECISION: ❌ REJECT</h3>

    <hr>

    <h4>Official Referee Report</h4>

    <p>Dear Authors,</p>

    <p>Thank you for submitting your manuscript, "Leveraging Soft Information for Economic Insights: An NLP Approach to the Beige Book,"
    to our journal. We appreciate your efforts and the innovative approach you have taken.</p>

    <p>The review panel has completed its assessment, and while the conceptual ambition of your work was noted, particularly in
    leveraging "soft information" from the Beige Book for regional insights and recession forecasting, the panel has identified
    <strong>fundamental and pervasive issues with the empirical execution and reporting</strong> that prevent the paper from meeting
    our journal's standards.</p>

    <p><strong>Specifically, the Mathematician identified critical issues:</strong></p>
    <ol>
        <li>Pervasive lack of explicit declaration regarding robust standard errors across all regression tables (Tables 1-6),
        rendering all reported statistical significance questionable.</li>
        <li>Absence of formal stationarity tests for time series variables, crucial for valid inference.</li>
        <li>Severe misrepresentation of Adjusted R² in Table 6, Model (1), where <strong>-0.111 was reported as "relatively high."</strong>
        This fundamentally undermines empirical evidence for regional analysis and casts doubt on overall empirical judgment.</li>
    </ol>

    <p><strong>The Historian reinforced these concerns:</strong></p>
    <p>While the paper excelled in contextualizing its work, the authors' misrepresentation of the negative Adjusted R² constituted
    an "egregious error" and a "critical breach of scientific integrity." This directly undermines the credibility of your claimed
    novelty in shedding light on regional economic activity, a core contribution you explicitly highlighted.</p>

    <p><strong>The Visionary acknowledged:</strong></p>
    <p>While recognizing the paper's "genuinely novel and field-changing idea" in systematically leveraging "soft information,"
    they concurred that the current empirical execution is insufficient to fully substantiate these groundbreaking claims.</p>

    <p style="margin-top: 20px;"><strong>Given the fundamental and pervasive nature of these empirical and methodological flaws,
    which include issues of scientific integrity in reporting findings, the paper cannot be accepted.</strong> The panel concluded
    that the empirical execution is "fundamentally unsound" and that a "complete re-evaluation and re-estimation of its empirical
    section" would be necessary, rendering the current manuscript unsuitable for publication.</p>

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
        - ✅ Historian shifted verdict after Mathematician's evidence (PASS → FAIL)
        - ✅ Cross-domain synthesis prevented single-perspective bias
        - ✅ No significant hallucinations detected in debate

        **Points for Consideration:**
        - ⚠️ Mathematician may be overly punitive on negative adjusted R² issue
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
