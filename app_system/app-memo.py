# app-memo.py - Memo Evaluation Agent
"""
Streamlit app for evaluating policy memos using Multi-Agent Debate.

This is a memo-specific version that uses memo evaluation personas
(Policy Analyst, Data Analyst, Stakeholder Analyst, Implementation Analyst,
Financial Stability Analyst) instead of academic research personas.
"""

import streamlit as st
import json
import asyncio
import datetime
import tempfile
import os
import zipfile
from io import BytesIO
import pdfplumber

from utils import cm
from referee.memo_engine import execute_debate_pipeline, MEMO_SYSTEM_PROMPTS
from referee._utils.summarizer import summarize_all_rounds

# Import helper functions from archived full output UI (domain-agnostic)
from referee._archived.full_output_ui import (
    CUSTOM_CSS,
    format_severity_labels,
    format_verdict,
    format_final_verdict,
    extract_verdict,
    summarize_text,
    format_round1_output,
    format_round2a_output,
    format_round2c_output,
    generate_summary_table
)

# Import equation/table fixer
from section_eval.region_fixer import render_region_fixer


# Custom CSS for visual enhancements
APP_CUSTOM_CSS = """
<style>
    .memo-type-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 5px;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .memo-type-card:hover {
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
    .analyst-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 5px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .analyst-card h4 {
        margin: 0 0 8px 0;
        color: white;
    }
    .analyst-card p {
        margin: 0;
        font-size: 13px;
        opacity: 0.95;
    }
</style>
"""


def _render_memo_architecture():
    """Display the Memo Evaluation architecture with visual design"""
    with st.expander("📐 **How It Works**", expanded=False):
        st.markdown('<div class="architecture-section">', unsafe_allow_html=True)
        st.markdown("#### Multi-Agent Debate for Policy Memos")

        # Step 1
        st.markdown("""
        <div class="step-box">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span class="step-number">1</span>
                <div style="margin-left: 20px; color: #333;">
                    <strong style="font-size: 18px;">Independent Evaluation</strong><br/>
                    <span style="font-size: 14px;">Each analyst evaluates independently and provides domain-specific verdict with evidence</span>
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
                    <span style="font-size: 14px;">Analysts read each other's reports, challenge assumptions, and engage in debate</span>
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
                <span style="margin-left: 10px; font-size: 13px; color: #555;">Analysts challenge each other's reasoning and synthesize insights</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="background: #9fa8da; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px;">2b</span>
                <strong style="margin-left: 10px; font-size: 15px;">Answer Questions</strong><br/>
                <span style="margin-left: 10px; font-size: 13px; color: #555;">Analysts respond to peer questions with evidence</span>
            </div>
            <div>
                <span style="background: #9fa8da; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px;">2c</span>
                <strong style="margin-left: 10px; font-size: 15px;">Final Amendments</strong><br/>
                <span style="margin-left: 10px; font-size: 13px; color: #555;">Analysts integrate peer feedback and submit final verdicts</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Step 3
        st.markdown("""
        <div class="step-box">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span class="step-number">3</span>
                <div style="margin-left: 20px; color: #333;">
                    <strong style="font-size: 18px;">Senior Reviewer Decision</strong><br/>
                    <span style="font-size: 14px;">Weighted consensus: PASS=1.0, REVISE=0.5, FAIL=0.0 → Final recommendation</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    """Extract text from uploaded file."""
    if filename.endswith('.txt'):
        return file_bytes.decode('utf-8', errors='replace')
    elif filename.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            with pdfplumber.open(tmp_path) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            return text
        finally:
            os.unlink(tmp_path)
    else:
        return file_bytes.decode('utf-8', errors='replace')


def main():
    st.set_page_config(layout="wide", page_title="Memo Evaluation Agent")

    # Apply custom CSS
    st.markdown(APP_CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("📝 Memo Evaluation Agent")
    st.markdown("This agent helps you evaluate policy memos using multi-agent debate with specialized analysts.")

    # Architecture display
    _render_memo_architecture()

    st.markdown("---")

    # File uploader
    with st.expander("Document Uploader", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload policy memos for evaluation.",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            key="memo_file_uploader"
        )

        # Store files in session state
        if "memo_files" not in st.session_state:
            st.session_state.memo_files = {}

        if uploaded_files:
            count = 0
            for uploaded_file in uploaded_files:
                st.session_state.memo_files[uploaded_file.name] = uploaded_file.getvalue()
                count += 1
            st.success(f"Successfully uploaded {count} file(s)")

        st.markdown("---")
        st.markdown("**Or paste memo text directly**")
        pasted_text = st.text_area(
            "Paste your memo text here:",
            height=200,
            key="memo_paste_text_area",
        )
        if st.button("Add pasted text", key="memo_paste_submit_btn"):
            if pasted_text.strip():
                name = "Pasted Memo.txt"
                st.session_state.memo_files[name] = pasted_text.encode("utf-8")
                st.success(f"Added pasted text as **{name}**")
            else:
                st.warning("Please paste some text first.")

    # Show available files
    if st.session_state.get("memo_files"):
        st.info(f"Available files: {', '.join(list(st.session_state.memo_files.keys()))}")

    st.markdown("---")

    # Memo type selection
    st.markdown("### Memo Type Selection (Optional)")
    st.markdown("Select your memo type to guide analyst selection.")

    memo_type_options = {
        "— Auto-detect —": None,
        "Policy Recommendation": "policy_recommendation",
        "Analytical Briefing": "analytical_briefing",
        "Decision Memo": "decision_memo",
        "Threat Assessment / Risk Briefing": "threat_assessment"
    }

    selected_memo_type_label = st.selectbox(
        "Memo Type",
        options=list(memo_type_options.keys()),
        key="memo_type_select",
        help="Choose the type that best describes your memo. This affects analyst selection."
    )

    memo_type = memo_type_options[selected_memo_type_label]

    if memo_type:
        descriptions = {
            "policy_recommendation": "Proposes specific policy actions and recommendations",
            "analytical_briefing": "Provides analysis of a situation without specific recommendations",
            "decision_memo": "Presents options and recommends a course of action",
            "threat_assessment": "Analyzes emerging threats, risks, or vulnerabilities (cybersecurity, tech risks, etc.)"
        }
        st.info(f"**{selected_memo_type_label}**: {descriptions.get(memo_type, '')}")

    st.markdown("---")

    # Analyst descriptions
    with st.expander("👥 **Available Analysts** (3 will be selected)", expanded=False):
        st.markdown("The system will automatically select the 3 most relevant analysts based on your memo content, or you can manually select them below.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="analyst-card">
                <h4>🏛️ Policy Analyst</h4>
                <p>Evaluates policy logic, recommendation soundness, and problem diagnosis</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="analyst-card">
                <h4>📊 Data Analyst</h4>
                <p>Evaluates evidence quality, data sources, and support for claims</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="analyst-card">
                <h4>👥 Stakeholder Analyst</h4>
                <p>Evaluates stakeholder identification, impact analysis, and equity considerations</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="analyst-card">
                <h4>⚙️ Implementation Analyst</h4>
                <p>Evaluates feasibility, action items, timelines, and practical execution</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="analyst-card">
                <h4>💰 Financial Stability Analyst</h4>
                <p>Evaluates costs, fiscal impact, economic risks, and financial stability</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Main evaluation interface
    st.subheader("🔍 Memo Evaluation")

    # File selection
    if not st.session_state.get("memo_files"):
        st.warning("Please upload or paste a memo above to begin evaluation.")
        return

    file_options = list(st.session_state.memo_files.keys())
    selected_file = st.selectbox("Select memo to evaluate:", file_options, key="selected_memo_file")

    # Manual analyst selection (optional)
    st.markdown("#### Analyst Selection (Optional)")
    col1, col2 = st.columns([2, 1])

    with col1:
        use_manual_selection = st.checkbox(
            "Manually select analysts (otherwise LLM auto-selects based on memo content)",
            key="use_manual_analyst_selection"
        )

    manual_personas = None
    manual_weights = None

    if use_manual_selection:
        st.markdown("##### Select 2-5 analysts:")

        all_analysts = list(MEMO_SYSTEM_PROMPTS.keys())
        selected_analysts = st.multiselect(
            "Analysts",
            options=all_analysts,
            default=["Policy Analyst", "Data Analyst", "Implementation Analyst"],
            key="manual_analyst_multiselect"
        )

        if len(selected_analysts) < 2:
            st.warning("Please select at least 2 analysts.")
        elif len(selected_analysts) > 5:
            st.warning("Please select at most 5 analysts.")
        else:
            manual_personas = selected_analysts

            # Optional: manual weights
            use_manual_weights = st.checkbox(
                "Manually specify weights (otherwise LLM assigns weights)",
                key="use_manual_weights"
            )

            if use_manual_weights:
                st.markdown("##### Assign weights (must sum to 1.0):")
                manual_weights = {}
                weight_cols = st.columns(len(manual_personas))

                for idx, analyst in enumerate(manual_personas):
                    with weight_cols[idx]:
                        weight = st.number_input(
                            analyst,
                            min_value=0.0,
                            max_value=1.0,
                            value=1.0/len(manual_personas),
                            step=0.05,
                            key=f"weight_{analyst}"
                        )
                        manual_weights[analyst] = weight

                total_weight = sum(manual_weights.values())
                if abs(total_weight - 1.0) > 0.01:
                    st.error(f"Weights must sum to 1.0 (current sum: {total_weight:.2f})")
                    manual_weights = None
                else:
                    st.success(f"✓ Weights sum to {total_weight:.2f}")

    # Custom evaluation context
    with st.expander("🎯 **Custom Evaluation Context** (Optional)", expanded=False):
        st.markdown("""
        Provide specific evaluation priorities or focus areas. Examples:
        - "Focus on political feasibility and stakeholder buy-in"
        - "Evaluate urgency and timeline realism"
        - "Check readiness for senior leadership review"
        - "Assess fiscal impact and budget implications"
        """)
        custom_context = st.text_area(
            "Evaluation priorities:",
            height=100,
            key="memo_custom_context"
        )
    custom_context = custom_context if custom_context and custom_context.strip() else None

    # Run evaluation button
    if st.button("🚀 Run Multi-Agent Evaluation", type="primary", key="run_memo_evaluation"):
        if not selected_file:
            st.error("Please select a file to evaluate.")
            return

        # Extract memo text
        memo_bytes = st.session_state.memo_files[selected_file]
        memo_text = extract_text_from_file(selected_file, memo_bytes)

        if not memo_text.strip():
            st.error("Could not extract text from the selected file.")
            return

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(message, progress):
            progress_bar.progress(progress)
            status_text.text(message)

        # Run debate pipeline
        with st.spinner("Running multi-agent evaluation..."):
            try:
                results = asyncio.run(execute_debate_pipeline(
                    memo_text,
                    progress_callback=update_progress,
                    memo_type=memo_type,
                    custom_context=custom_context,
                    manual_personas=manual_personas,
                    manual_weights=manual_weights
                ))

                # Store results in session state
                st.session_state.memo_evaluation_results = results

                st.success("✅ Evaluation complete!")
                progress_bar.empty()
                status_text.empty()

            except Exception as e:
                st.error(f"Error during evaluation: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Display results if available
    if st.session_state.get("memo_evaluation_results"):
        results = st.session_state.memo_evaluation_results

        st.markdown("---")
        st.markdown("## 📊 Evaluation Results")

        # Metadata
        with st.expander("ℹ️ **Evaluation Metadata**", expanded=False):
            metadata = results.get('metadata', {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Runtime", metadata.get('total_runtime_formatted', 'N/A'))
            with col2:
                st.metric("Total Cost", f"${metadata.get('token_usage', {}).get('cost_usd', {}).get('total', 0):.4f}")
            with col3:
                st.metric("Total Tokens", f"{metadata.get('token_usage', {}).get('total_tokens', 0):,}")

        # Round 0: Analyst Selection
        st.markdown("### Round 0: Analyst Selection")
        selection_data = results.get('round_0', {})
        selected_analysts = selection_data.get('selected_personas', [])
        weights = selection_data.get('weights', {})

        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**Selected Analysts:** {', '.join(selected_analysts)}")
            st.write(f"**Justification:** {selection_data.get('justification', 'N/A')}")
        with col2:
            st.write("**Weights:**")
            for analyst, weight in weights.items():
                st.write(f"- {analyst}: {weight:.2f}")

        # Summarize all rounds using LLM
        st.markdown("### 🔄 Generating Summaries...")
        with st.spinner("Summarizing debate rounds with LLM..."):
            try:
                summaries_raw = asyncio.run(summarize_all_rounds(results))
                # Map summary keys to match round names
                summaries = {
                    'round_1': summaries_raw.get('round_1_summaries', {}),
                    'round_2a': summaries_raw.get('round_2a_summaries', {}),
                    'round_2b': summaries_raw.get('round_2b_summaries', {}),
                    'round_2c': summaries_raw.get('round_2c_summaries', {}),
                    'final_decision': summaries_raw.get('editor_summary', '')
                }
            except Exception as e:
                st.error(f"Error generating summaries: {e}")
                summaries = {}

        # Display summarized rounds
        for round_name in ['round_1', 'round_2a', 'round_2b', 'round_2c']:
            round_label = {
                'round_1': 'Round 1: Independent Evaluation',
                'round_2a': 'Round 2A: Cross-Examination',
                'round_2b': 'Round 2B: Answer Questions',
                'round_2c': 'Round 2C: Final Amendments'
            }[round_name]

            st.markdown(f"### {round_label}")

            if round_name in summaries and summaries[round_name]:
                # Show summarized version
                for analyst, summary in summaries[round_name].items():
                    with st.expander(f"**{analyst}** (Summarized)", expanded=False):
                        # Handle both string and dict summaries (round_1 and round_2c return dicts)
                        if isinstance(summary, dict):
                            st.markdown(summary.get('summary', str(summary)))
                        else:
                            st.markdown(summary)

                        # Toggle to show full report
                        show_full_key = f"show_full_{round_name}_{analyst}"
                        if st.checkbox("📄 View Full Report", key=show_full_key):
                            full_report = results.get(round_name, {}).get(analyst, "N/A")
                            st.markdown("---")
                            st.markdown("**Full Report:**")
                            st.markdown(full_report)
            else:
                # Fallback to full reports
                for analyst, report in results.get(round_name, {}).items():
                    with st.expander(f"**{analyst}**", expanded=False):
                        st.markdown(report)

        # Consensus calculation
        st.markdown("### 🎯 Consensus & Final Decision")
        consensus = results.get('consensus', {})

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Weighted Consensus Score", f"{consensus.get('weighted_score', 0):.3f}")
            st.write("**Individual Verdicts:**")
            for analyst, verdict in consensus.get('verdicts', {}).items():
                st.write(f"- {analyst}: **{verdict}**")

        with col2:
            decision = consensus.get('decision', 'UNKNOWN')
            if decision == "ACCEPT":
                st.success(f"**Final Decision:** {decision}")
            elif decision == "REJECT AND RESUBMIT":
                st.warning(f"**Final Decision:** {decision}")
            else:
                st.error(f"**Final Decision:** {decision}")

        # Final reviewer report
        st.markdown("### 📋 Final Reviewer Report")
        final_decision = results.get('final_decision', 'N/A')

        # Try to summarize the final decision
        if 'final_decision' in summaries and summaries['final_decision']:
            with st.expander("**Senior Reviewer Summary** (Summarized)", expanded=True):
                st.markdown(summaries['final_decision'])

                # Toggle to show full report
                if st.checkbox("📄 View Full Report", key="show_full_final_decision"):
                    st.markdown("---")
                    st.markdown("**Full Report:**")
                    st.markdown(final_decision)
        else:
            with st.expander("**Senior Reviewer Report**", expanded=True):
                st.markdown(final_decision)

        # Download options
        st.markdown("---")
        st.markdown("### 💾 Download Results")

        col1, col2 = st.columns(2)

        with col1:
            # JSON download
            json_str = json.dumps(results, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download Full Results (JSON)",
                data=json_str,
                file_name=f"memo_evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_memo_json"
            )

        with col2:
            # Summary report download
            summary_report = f"""# Memo Evaluation Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Selected Analysts
{', '.join(selected_analysts)}

Weights: {', '.join([f'{a}: {w:.2f}' for a, w in weights.items()])}

## Consensus Score
{consensus.get('weighted_score', 0):.3f}

## Final Decision
{decision}

## Individual Verdicts
{chr(10).join([f'- {a}: {v}' for a, v in consensus.get('verdicts', {}).items()])}

## Senior Reviewer Report
{final_decision}
"""
            st.download_button(
                label="📥 Download Summary Report (MD)",
                data=summary_report,
                file_name=f"memo_evaluation_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key="download_memo_summary"
            )

    # Footer
    st.markdown("---")
    st.caption("Memo Evaluation Agent powered by Multi-Agent Debate")


if __name__ == "__main__":
    main()
