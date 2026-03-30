# app_full_output.py - Demo with full verbose output (archived version)
"""
This demo uses the archived full-output UI that shows uncompressed debate results.
For production use, see the main app.py which uses the cleaner summarized version.
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path so we can import from app_system
sys.path.insert(0, str(Path(__file__).parent.parent))

from referee._archived import RefereeReportChecker
from section_eval import SectionEvaluatorApp
from utils import cm

WORKFLOWS = {
    "Referee Report (Full Output)": RefereeReportChecker,
    "Section Evaluator": SectionEvaluatorApp,
}

# Custom CSS for visual enhancements
CUSTOM_CSS = """
<style>
    .paper-type-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 5px;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .paper-type-card:hover {
        transform: translateY(-2px);
    }
    .step-box {
        background: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 10px 0;
    }
    .step-number {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 16px;
        min-width: 40px;
        text-align: center;
        display: inline-block;
    }
    .architecture-section {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
    }
</style>
"""

def _ensure_session_keys():
    """Create session keys that the workflows expect."""
    if "file_data" not in st.session_state:
        st.session_state.file_data = {}
    # Track which tab was active last (for LLM history clearing)
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = None

def _safe_render(workflow_instance, files):
    """
    Render the workflow UI with the shared files
    """
    try:
        workflow_instance.render_ui(files=files)
    except Exception as e:
        st.error(f"Error rendering workflow UI: {e}")

def _safe_render_with_paper_type(workflow_instance, files):
    """
    Render the Section Evaluator workflow UI with paper type
    """
    try:
        workflow_instance.render_ui(files=files, paper_type=st.session_state.get("paper_type"))
    except Exception as e:
        st.error(f"Error rendering workflow UI: {e}")

def _render_section_evaluator_architecture():
    """Display the Section Evaluator architecture with visual design"""
    with st.expander("📐 **How It Works**", expanded=False):
        st.markdown('<div class="architecture-section">', unsafe_allow_html=True)
        st.markdown("#### Paper-Type-Aware Evaluation Framework")

        # Step 1
        st.markdown("""
        <div class="step-box">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span class="step-number">1</span>
                <div style="margin-left: 15px; color: #333;">
                    <strong style="font-size: 16px;">Text Extraction & Section Detection</strong><br/>
                    <span style="font-size: 13px;">PDF/LaTeX parsing → Hierarchical section detection</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Step 2
        st.markdown("""
        <div class="step-box">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span class="step-number">2</span>
                <div style="margin-left: 15px; color: #333;">
                    <strong style="font-size: 16px;">Paper-Type-Specific Criteria Mapping</strong><br/>
                    <span style="font-size: 13px;">Different sections and criteria based on paper type</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Paper type comparison
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
                <p style="font-size: 11px; margin-top: 10px; color: #666;">Emphasizes identification strategy</p>
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
                <p style="font-size: 11px; margin-top: 10px; color: #666;">Emphasizes mathematical rigor</p>
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
                <p style="font-size: 11px; margin-top: 10px; color: #666;">Emphasizes practical applicability</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<p style='text-align: center; font-size: 12px; color: #666; margin-top: 10px;'>⭐ = Highest importance multiplier</p>", unsafe_allow_html=True)

        # Step 3
        st.markdown("""
        <div class="step-box">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span class="step-number">3</span>
                <div style="margin-left: 15px; color: #333;">
                    <strong style="font-size: 16px;">Weighted Section Evaluation</strong><br/>
                    <span style="font-size: 13px;">Per-criterion scoring (1-5) × Criterion weights → Adjusted Score</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Step 4
        st.markdown("""
        <div class="step-box">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span class="step-number">4</span>
                <div style="margin-left: 15px; color: #333;">
                    <strong style="font-size: 16px;">Overall Scoring & Feedback</strong><br/>
                    <span style="font-size: 13px;">Publication readiness + Actionable improvements per section</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

def _render_referee_report_architecture():
    """Display the Referee Report architecture with visual design"""
    with st.expander("📐 **How It Works**", expanded=False):
        st.markdown('<div class="architecture-section">', unsafe_allow_html=True)
        st.markdown("#### Multi-Agent Debate Architecture")

        # Step 1
        st.markdown("""
        <div class="step-box">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span class="step-number">1</span>
                <div style="margin-left: 20px; color: #333;">
                    <strong style="font-size: 18px;">Independent Evaluation</strong><br/>
                    <span style="font-size: 14px;">Each agent evaluates independently and provides domain-specific verdict with evidence</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Step 2
        st.markdown("""
        <div class="step-box">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span class="step-number">2</span>
                <div style="margin-left: 20px; color: #333;">
                    <strong style="font-size: 18px;">Cross-Examination</strong><br/>
                    <span style="font-size: 14px;">Agents read each other's reports, challenge assumptions, and engage in debate</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Sub-steps
        st.markdown("""
        <div style="background: #f8f9fa; padding: 15px; padding-left: 40px; border-radius: 8px; border-left: 4px solid #9fa8da; margin: 10px 0 10px 40px;">
            <div style="margin-bottom: 12px;">
                <span style="background: #9fa8da; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px;">2a</span>
                <strong style="margin-left: 10px; font-size: 15px;">Debate Round</strong><br/>
                <span style="margin-left: 10px; font-size: 13px; color: #555;">Agents challenge each other's reasoning and synthesize insights</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="background: #9fa8da; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px;">2b</span>
                <strong style="margin-left: 10px; font-size: 15px;">Answer Questions</strong><br/>
                <span style="margin-left: 10px; font-size: 13px; color: #555;">Agents respond to peer questions with evidence</span>
            </div>
            <div>
                <span style="background: #9fa8da; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px;">2c</span>
                <strong style="margin-left: 10px; font-size: 15px;">Final Amendments</strong><br/>
                <span style="margin-left: 10px; font-size: 13px; color: #555;">Agents integrate peer feedback and submit final verdicts</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Step 3
        st.markdown("""
        <div class="step-box">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span class="step-number">3</span>
                <div style="margin-left: 20px; color: #333;">
                    <strong style="font-size: 18px;">Editor Decision</strong><br/>
                    <span style="font-size: 14px;">Weighted consensus: PASS=1.0, REVISE=0.5, FAIL=0.0 → Final recommendation</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide", page_title="Evaluation Agent")

    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("Evaluation Agent")

    # Use cleaner heading style
    st.markdown("This agent helps you evaluate and improve your academic work. Choose one of two workflows below to get started.")

    _ensure_session_keys()

    # Initialize paper type in session state
    if "paper_type" not in st.session_state:
        st.session_state.paper_type = None

    # ----------------- Paper Type Selection -----------------
    st.markdown("### 📋 Select Paper Type")
    st.markdown("Choose the type that best matches your manuscript (affects Section Evaluator criteria):")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Empirical", use_container_width=True, type="primary" if st.session_state.paper_type == "empirical" else "secondary"):
            st.session_state.paper_type = "empirical"

    with col2:
        if st.button("📐 Theoretical", use_container_width=True, type="primary" if st.session_state.paper_type == "theoretical" else "secondary"):
            st.session_state.paper_type = "theoretical"

    with col3:
        if st.button("🏛️ Policy", use_container_width=True, type="primary" if st.session_state.paper_type == "policy" else "secondary"):
            st.session_state.paper_type = "policy"

    if st.session_state.paper_type:
        paper_type_display = st.session_state.paper_type.capitalize()

        # Color-coded display based on paper type
        if st.session_state.paper_type == "empirical":
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 8px; border: 3px solid #003d82; margin: 10px 0;">
                <span style="color: #003d82; font-weight: bold; font-size: 16px;">📊 Selected: {paper_type_display}</span>
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.paper_type == "theoretical":
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 8px; border: 3px solid #1b5e20; margin: 10px 0;">
                <span style="color: #1b5e20; font-weight: bold; font-size: 16px;">📐 Selected: {paper_type_display}</span>
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.paper_type == "policy":
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 8px; border: 3px solid #c62828; margin: 10px 0;">
                <span style="color: #c62828; font-weight: bold; font-size: 16px;">🏛️ Selected: {paper_type_display}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("💡 Select a paper type above to customize the evaluation criteria")

    st.markdown("---")

    # ----------------- Shared File Uploader -----------------
    with st.expander("Document Uploader", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload manuscripts and/or referee reports for evaluation.",
            type=['pdf', 'txt', 'docx', 'tex'],
            accept_multiple_files=True
        )
        if uploaded_files:
            count = 0
            for uploaded_file in uploaded_files:
                st.session_state.file_data[uploaded_file.name] = uploaded_file.getvalue()
                count += 1
            st.success(f"Successfully uploaded {count} file(s)")
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
                st.success(f"Added pasted text as **{name}**")
                try:
                    cm.clear_history()
                except Exception:
                    pass
            else:
                st.warning("Please paste some text first.")

    # Show available files in compact form
    if st.session_state.file_data:
        st.info(f"Available files: {', '.join(list(st.session_state.file_data.keys()))}")

    # ----------------- Create visual tabs -----------------
    tab_labels = list(WORKFLOWS.keys())
    tab_objs = st.tabs(tab_labels)
    tab_map = dict(zip(tab_labels, tab_objs))

    # Make tabs larger
    css = '''
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.25rem;
        }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

    # ----------------- Render each tab -----------------
    for label, tab in tab_map.items():
        with tab:
            # Clear LLM history when switching tabs
            if st.session_state.active_tab != label:
                if st.session_state.active_tab is not None:
                    try:
                        cm.clear_history()
                    except Exception:
                        pass
                st.session_state.active_tab = label

            # Show architecture display specific to each workflow
            if label == "Section Evaluator":
                _render_section_evaluator_architecture()
            elif label == "Referee Report":
                _render_referee_report_architecture()

            st.markdown("---")

            # Workflow session key (persist instance)
            key = f"{label.lower().replace(' ', '_')}_workflow"
            if key not in st.session_state:
                # Instantiate and store
                try:
                    st.session_state[key] = WORKFLOWS[label]()
                except Exception as e:
                    st.error(f"Failed to initialize {label} workflow: {e}")
                    continue

            instance = st.session_state[key]

            # Safe render with shared files (pass paper_type for Section Evaluator)
            if label == "Section Evaluator":
                _safe_render_with_paper_type(instance, files=st.session_state.get("file_data", {}))
            else:
                _safe_render(instance, files=st.session_state.get("file_data", {}))

    # Footer
    st.markdown("---")
    st.caption("Choose a tab above to access different evaluation workflows.")

if __name__ == "__main__":
    main()