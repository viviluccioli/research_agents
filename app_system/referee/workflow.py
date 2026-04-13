# workflow.py - Main Multi-Agent Referee Report UI
"""
Main production UI for the multi-agent debate (MAD) referee report system.

This is the official workflow used in app.py. It uses LLM-powered summarization
to compress debate outputs for cleaner display while preserving full reports
in expandable sections.
"""
import streamlit as st
import json
import tempfile
import os
import asyncio
import datetime
import re
import zipfile
from typing import Dict, List, Any
from io import BytesIO
import pdfplumber
import pandas as pd
from utils import cm, single_query
from referee.engine import execute_debate_pipeline, SELECTION_PROMPT, SYSTEM_PROMPTS, DEBATE_PROMPTS
from referee._utils.summarizer import summarize_all_rounds

# Import helper functions from archived full output UI
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


class RefereeWorkflow:
    """
    Main production workflow for multi-agent debate paper evaluation.

    This is the official UI used in app.py. It uses LLM-powered summarization
    to compress debate outputs for cleaner display while preserving full reports
    in expandable sections.
    """

    def __init__(self, llm=cm):
        """Initialize with conversation manager from utils."""
        self.llm = llm

    def render_ui(self, files=None):
        """Render the Streamlit UI for the multi-agent referee report."""
        # Apply custom CSS
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

        st.subheader("Multi-Agent Referee Evaluation")
        st.info("📊 **NOTE**: This version uses additional LLM passes to compress outputs for cleaner display.")
        st.write("This tool uses a multi-agent debate system to evaluate your manuscript from multiple perspectives.")

        # Add description expander with personas
        with st.expander("📖 **How This System Works**", expanded=False):
            st.markdown("""
### What This System Does

The **Multi-Agent Debate (MAD) System** evaluates research papers through structured debate between specialized AI personas.

### Workflow Overview

The system operates in **5 sequential rounds**:

1. **Round 0 - Persona Selection**: Select 2-5 personas either automatically (LLM Chief Editor chooses based on paper content and type) or manually. Importance weights are assigned (summing to 1.0), either automatically or manually.

2. **Round 1 - Independent Evaluation**: Each selected persona independently evaluates the paper from their domain expertise, identifying flaws with severity labels ([FATAL], [MAJOR], [MINOR]) and providing an initial verdict (PASS/REVISE/FAIL).

3. **Round 2A - Cross-Examination**: Personas read each other's Round 1 evaluations and engage in cross-domain synthesis, constructive pushback, and clarification questions.

4. **Round 2B - Direct Examination**: Each persona responds to questions directed at them from Round 2A, providing evidence and taking a position (CONCEDE or DEFEND).

5. **Round 2C - Final Amendments**: After reviewing the full debate transcript, each persona submits a final amended verdict with justification.

6. **Round 3 - Editor Consensus**: A weighted consensus score is computed mathematically (PASS=1.0, REVISE=0.5, FAIL=0.0), and the Editor writes the official referee report.

### Decision Thresholds

- **Consensus Score > 0.75**: ACCEPT
- **Consensus Score 0.40–0.75**: REJECT AND RESUBMIT
- **Consensus Score < 0.40**: REJECT

---

### 👥 Available Personas (2-5 will be selected)
            """)

            # Compact horizontal persona cards
            st.markdown("""
            <style>
            .persona-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
                padding: 15px;
                margin: 5px;
                text-align: center;
                color: white;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .persona-card h4 {
                margin: 0 0 8px 0;
                font-size: 18px;
            }
            .persona-card p {
                margin: 4px 0;
                font-size: 12px;
                opacity: 0.95;
            }
            .persona-theorist { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .persona-empiricist { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
            .persona-historian { background: linear-gradient(135deg, #7f7fd5 0%, #86a8e7 100%); }
            .persona-visionary { background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%); }
            .persona-policymaker { background: linear-gradient(135deg, #ee5a6f 0%, #f29263 100%); }
            </style>
            <div style="display: flex; justify-content: space-between; gap: 8px; margin: 15px 0;">
                <div class="persona-card persona-theorist">
                    <h4>🔢 Theorist</h4>
                    <p><strong>Focus:</strong> Mathematical logic & proofs</p>
                </div>
                <div class="persona-card persona-empiricist">
                    <h4>📊 Empiricist</h4>
                    <p><strong>Focus:</strong> Data & identification</p>
                </div>
                <div class="persona-card persona-historian">
                    <h4>📚 Historian</h4>
                    <p><strong>Focus:</strong> Literature context</p>
                </div>
                <div class="persona-card persona-visionary">
                    <h4>🚀 Visionary</h4>
                    <p><strong>Focus:</strong> Innovation & novelty</p>
                </div>
                <div class="persona-card persona-policymaker">
                    <h4>🏛️ Policymaker</h4>
                    <p><strong>Focus:</strong> Policy relevance</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.info("💡 You can choose personas and weights automatically (system-driven) or manually configure them below.")

        st.markdown("---")

        # Check if files are available
        if not files:
            st.info("Please upload your manuscript using the document uploader above.")
            return

        # File selection
        manuscript_file = st.selectbox(
            "Select your manuscript for multi-agent evaluation",
            options=list(files.keys()),
            key="manuscript_selector"
        )

        # Output mode selection
        st.markdown("---")
        st.markdown("#### ⚙️ Output Settings")

        use_summarizer = st.checkbox(
            "📊 Use LLM Summarizer (cleaner display, more API calls)",
            value=False,
            help="When enabled: Adds ~10-15 extra API calls to compress debate outputs for cleaner display.\n"
                 "When disabled: Shows full outputs only, 14 API calls total (3 personas × 4 rounds + 2 editor)."
        )

        if use_summarizer:
            st.info("💡 **Mode**: Summarized display (adds ~10-15 API calls for compression)")
        else:
            st.success("⚡ **Mode**: Full output only (14 API calls total - most cost-efficient)")

        st.markdown("---")

        # ==================== PERSONA SELECTION & CUSTOM CONTEXT ====================
        st.markdown("#### 🎭 Persona Selection & Evaluation Context")

        # Get paper type from global session state
        paper_type = st.session_state.get("paper_type")
        if paper_type:
            st.success(f"📄 **Paper Type**: {paper_type.title()} — This will guide persona selection")
        else:
            st.warning("⚠️ No paper type selected. Please select a paper type above for better persona recommendations.")

        # Custom context text box
        custom_context = st.text_area(
            "**Optional: Additional Evaluation Context** (Tell us what you're looking for)",
            placeholder="Example: 'Focus on statistical rigor and replicability' or 'Evaluate suitability for submission to top field journal' or 'Check if methods are appropriate for policy analysis'",
            height=100,
            help="Provide any specific evaluation priorities, focus areas, or context that should guide the review. This will be considered throughout the entire debate process.",
            key="custom_context_input"
        )

        st.markdown("---")
        st.markdown("**Persona Selection Mode**")

        persona_mode = st.radio(
            "How should personas be selected?",
            [
                "🤖 Fully Automatic (System chooses personas and weights)",
                "🔢 Specify Count (You choose how many personas, system selects which ones)",
                "🎯 Manual Selection (You choose personas, system assigns weights)",
                "⚙️ Full Manual (You choose personas AND weights)"
            ],
            key="persona_selection_mode",
            help="Choose how much control you want over persona selection. The paper type (if selected) will guide automatic selections."
        )

        # Initialize variables for persona selection
        manual_personas = None
        manual_weights = None

        # Handle different persona selection modes
        if persona_mode == "🔢 Specify Count (You choose how many personas, system selects which ones)":
            num_personas = st.slider(
                "How many personas should evaluate the paper?",
                min_value=2,
                max_value=5,
                value=3,
                key="num_personas_slider"
            )
            st.info(f"System will automatically select {num_personas} personas based on paper content" + (f" and paper type ({paper_type})" if paper_type else ""))

            # For this mode, we'll need to update the engine to support persona count
            # For now, we'll use manual_personas with None values to signal count-based selection
            # This requires additional engine updates - for now default to auto
            st.warning("⚠️ Count-based selection will default to automatic 3-persona selection. Full count control coming soon.")

        elif persona_mode == "🎯 Manual Selection (You choose personas, system assigns weights)":
            st.markdown("**Select 2-5 personas to evaluate your paper:**")
            available_personas = ["Theorist", "Empiricist", "Historian", "Visionary", "Policymaker"]

            cols = st.columns(5)
            selected_personas_list = []

            for idx, persona in enumerate(available_personas):
                with cols[idx]:
                    if st.checkbox(persona, key=f"persona_check_{persona}"):
                        selected_personas_list.append(persona)

            if len(selected_personas_list) < 2:
                st.warning("⚠️ Please select at least 2 personas")
            elif len(selected_personas_list) > 5:
                st.error("❌ Maximum 5 personas allowed")
            else:
                manual_personas = selected_personas_list
                st.success(f"✅ Selected {len(manual_personas)} personas: {', '.join(manual_personas)}")
                st.info("The system will automatically assign importance weights to these personas based on the paper content" + (f" and paper type ({paper_type})" if paper_type else ""))

        elif persona_mode == "⚙️ Full Manual (You choose personas AND weights)":
            st.markdown("**Select 2-5 personas and assign their weights:**")
            available_personas = ["Theorist", "Empiricist", "Historian", "Visionary", "Policymaker"]

            st.info("💡 Weights must sum to 1.0. Higher weight = more influence on final decision")

            selected_personas_dict = {}
            cols_check = st.columns(5)
            selected_for_weight = []

            # First row: checkboxes
            for idx, persona in enumerate(available_personas):
                with cols_check[idx]:
                    if st.checkbox(persona, key=f"persona_check_manual_{persona}"):
                        selected_for_weight.append(persona)

            if len(selected_for_weight) < 2:
                st.warning("⚠️ Please select at least 2 personas")
            elif len(selected_for_weight) > 5:
                st.error("❌ Maximum 5 personas allowed")
            else:
                # Second section: weight sliders
                st.markdown("**Assign weights (must sum to 1.0):**")
                cols_weight = st.columns(len(selected_for_weight))

                temp_weights = {}
                for idx, persona in enumerate(selected_for_weight):
                    with cols_weight[idx]:
                        default_weight = 1.0 / len(selected_for_weight)
                        weight = st.number_input(
                            persona,
                            min_value=0.0,
                            max_value=1.0,
                            value=default_weight,
                            step=0.05,
                            key=f"weight_{persona}"
                        )
                        temp_weights[persona] = weight

                weight_sum = sum(temp_weights.values())
                st.metric("Weight Sum", f"{weight_sum:.2f}", delta=f"{weight_sum - 1.0:.2f}" if abs(weight_sum - 1.0) > 0.01 else "✓")

                if abs(weight_sum - 1.0) < 0.01:  # Allow small floating point errors
                    manual_personas = selected_for_weight
                    manual_weights = temp_weights
                    st.success(f"✅ Weights configured: {', '.join([f'{p} ({w:.2f})' for p, w in manual_weights.items()])}")
                else:
                    st.error(f"❌ Weights must sum to 1.0 (current sum: {weight_sum:.2f})")

        else:  # Fully Automatic
            st.info("🤖 The system will automatically select 3 personas and assign weights based on paper content" + (f" and paper type ({paper_type})" if paper_type else ""))

        st.markdown("---")

        # Session state keys for this workflow
        extraction_key = f"referee_extraction_done_{manuscript_file}"
        fixes_key = f"referee_fixes_applied_{manuscript_file}"
        text_key = f"referee_text_{manuscript_file}"
        fixed_text_key = f"referee_fixed_text_{manuscript_file}"

        # Check workflow state
        extraction_done = st.session_state.get(extraction_key, False)
        fixes_applied = st.session_state.get(fixes_key, False)

        # Stage 1: Initial button to extract text
        if not extraction_done:
            st.info("👉 **Next Step**: Extract text from your PDF and review equations/tables before running the debate")
            if st.button("🚀 Step 1: Extract & Review Text", type="primary", use_container_width=True):
                if not manuscript_file:
                    st.error("Please select a manuscript file.")
                    return

                with st.spinner("Extracting text from manuscript..."):
                    # Extract text from manuscript
                    paper_text = self.extract_text_from_pdf(files[manuscript_file])
                    st.session_state[text_key] = paper_text
                    st.session_state[extraction_key] = True
                    st.success("✅ Text extracted! Review equations/tables below.")
                    st.rerun()

        # Stage 2: Show equation fixer UI (after extraction, before fixes applied)
        if extraction_done and not fixes_applied:
            paper_text = st.session_state.get(text_key, "")

            # Workflow progress indicator
            st.markdown("### 📍 Two-Stage Workflow")
            col_stage1, col_arrow, col_stage2 = st.columns([1, 0.2, 1])

            with col_stage1:
                st.markdown(
                    '<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); '
                    'color: white; padding: 15px; border-radius: 10px; text-align: center;">'
                    '<strong>STAGE 1: Fix Equations & Tables</strong><br/>'
                    '<span style="font-size: 12px;">Currently here ✓</span>'
                    '</div>',
                    unsafe_allow_html=True
                )

            with col_arrow:
                st.markdown('<div style="text-align: center; font-size: 30px;">→</div>', unsafe_allow_html=True)

            with col_stage2:
                st.markdown(
                    '<div style="background: #e0e0e0; color: #666; padding: 15px; '
                    'border-radius: 10px; text-align: center; border: 2px dashed #999;">'
                    '<strong>STAGE 2: Run Multi-Agent Debate</strong><br/>'
                    '<span style="font-size: 12px;">After fixing</span>'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.markdown("### 🔍 Extraction Quality Check")

            # Render the region fixer
            render_region_fixer(
                text=paper_text,
                manuscript_name=manuscript_file,
                cache_prefix="referee",
                llm_query_fn=single_query,
                min_confidence=0.5
            )

            # Don't proceed until fixes are applied
            return

        # Stage 3: Run the debate (after fixes applied)
        if extraction_done and fixes_applied:
            # Get the fixed text
            paper_text = st.session_state.get(fixed_text_key, st.session_state.get(text_key, ""))

            # Workflow progress indicator for stage 2
            st.markdown("### 📍 Two-Stage Workflow")
            col_stage1, col_arrow, col_stage2 = st.columns([1, 0.2, 1])

            with col_stage1:
                st.markdown(
                    '<div style="background: #4caf50; color: white; padding: 15px; '
                    'border-radius: 10px; text-align: center;">'
                    '<strong>STAGE 1: Fix Equations & Tables</strong><br/>'
                    '<span style="font-size: 12px;">✓ Complete</span>'
                    '</div>',
                    unsafe_allow_html=True
                )

            with col_arrow:
                st.markdown('<div style="text-align: center; font-size: 30px;">→</div>', unsafe_allow_html=True)

            with col_stage2:
                st.markdown(
                    '<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); '
                    'color: white; padding: 15px; border-radius: 10px; text-align: center;">'
                    '<strong>STAGE 2: Run Multi-Agent Debate</strong><br/>'
                    '<span style="font-size: 12px;">Ready to run ✓</span>'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # Cache controls
            st.markdown("#### 💾 Cache Settings")

            col_cache, col_clear = st.columns([3, 1])

            with col_cache:
                use_cache = st.checkbox(
                    "📦 Use cached results (if available)",
                    value=True,
                    help="Enable caching to reuse results from previous runs with the same paper and configuration. Can save $1.50-2.00 per run."
                )

                force_refresh = st.checkbox(
                    "🔄 Force refresh (ignore cache)",
                    value=False,
                    help="Run all rounds fresh even if cached results exist. Useful for testing prompt changes.",
                    disabled=not use_cache
                )

            with col_clear:
                from referee._utils.cache import compute_cache_key, clear_cache_for_paper, get_cache_stats

                # Compute cache key for this paper
                cache_key = compute_cache_key(
                    paper_text=paper_text,
                    selected_personas=manual_personas,
                    weights=manual_weights,
                    model_name="claude-3-7-sonnet",  # Approximate from config
                    paper_type=paper_type,
                    custom_context=custom_context if custom_context and custom_context.strip() else None
                )

                if st.button("🗑️ Clear Cache", help="Clear cached results for this specific paper", use_container_width=True):
                    if clear_cache_for_paper(cache_key):
                        st.success("✅ Cache cleared!")
                    else:
                        st.info("ℹ️ No cache to clear")
                    st.rerun()

            # Show cache stats
            if use_cache:
                from referee._utils.cache import check_cache_status
                cache_status = check_cache_status(cache_key)
                cached_rounds = [k for k, v in cache_status.items() if v]

                if cached_rounds:
                    st.info(f"💾 **Cache available** for: {', '.join(cached_rounds)}")
                else:
                    st.info("💾 **No cached results** - first run will be cached for future use")

            # Show global cache statistics
            with st.expander("📊 Global Cache Statistics"):
                stats = get_cache_stats()
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Cached Papers", stats['total_entries'])
                with col_stat2:
                    st.metric("Cache Size", f"{stats['total_size_mb']} MB")
                with col_stat3:
                    st.metric("Cache Location", "📁")

                st.caption(f"Location: `{stats['cache_dir']}`")

                from referee._utils.cache import clear_cache
                col_clear_old, col_clear_all = st.columns(2)
                with col_clear_old:
                    if st.button("🧹 Clear Old Cache (>30 days)", use_container_width=True):
                        removed, total = clear_cache(older_than_days=30)
                        st.success(f"✅ Removed {removed}/{total} old entries")
                        st.rerun()
                with col_clear_all:
                    if st.button("⚠️ Clear All Cache", use_container_width=True, type="secondary"):
                        removed, total = clear_cache(older_than_days=0)
                        st.success(f"✅ Removed all {removed} entries")
                        st.rerun()

            st.markdown("---")

            col_run, col_reset = st.columns([3, 1])

            with col_run:
                run_debate = st.button("🚀 Step 2: Run Multi-Agent Evaluation", type="primary", use_container_width=True)

            with col_reset:
                if st.button("🔄 Start Over", use_container_width=True):
                    # Clear all session state for this manuscript
                    st.session_state.pop(extraction_key, None)
                    st.session_state.pop(fixes_key, None)
                    st.session_state.pop(text_key, None)
                    st.session_state.pop(fixed_text_key, None)
                    st.session_state.pop(f"referee_region_fixes_{manuscript_file}", None)
                    st.info("Workflow reset. Click 'Step 1' to start over.")
                    st.rerun()

            if run_debate:
                # Run the multi-agent debate
                try:
                    # Create progress tracking
                    progress_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(stage, progress):
                        status_text.text(f"🔄 {stage}")
                        progress_bar.progress(progress)

                    # Execute debate pipeline with enhanced context
                    with st.spinner("Running multi-agent debate..."):
                        debate_results = asyncio.run(
                            execute_debate_pipeline(
                                paper_text,
                                progress_callback=update_progress,
                                paper_type=paper_type,
                                custom_context=custom_context if custom_context and custom_context.strip() else None,
                                manual_personas=manual_personas,
                                manual_weights=manual_weights,
                                use_cache=use_cache,
                                force_refresh=force_refresh
                            )
                        )

                    # Run summarization pass (optional)
                    if use_summarizer:
                        with st.spinner("Compressing outputs for display (additional LLM pass)..."):
                            summaries = asyncio.run(summarize_all_rounds(debate_results))
                    else:
                        # Skip summarization - use empty summaries dict
                        summaries = {
                            'round_1_summaries': {},
                            'round_2a_summaries': {},
                            'round_2b_summaries': {},
                            'round_2c_summaries': {},
                            'editor_summary': None
                        }
                        st.info("⚡ Skipped summarization - showing full outputs only")

                    # Store results in session state
                    st.session_state.debate_results = debate_results
                    st.session_state.debate_summaries = summaries
                    st.session_state.manuscript_file = manuscript_file  # Store filename for downloads
                    st.session_state.use_summarizer = use_summarizer  # Store for display

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                    # Display results
                    self.display_debate_results(debate_results, summaries)

                except Exception as e:
                    st.error(f"Error during multi-agent evaluation: {e}")
                    st.exception(e)


    def extract_text_from_pdf(self, file_content):
        """
        Extract text and tables from a PDF file.

        Returns:
            str: Extracted text with tables formatted as markdown
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()

            text = ""
            total_tables = 0
            total_chars = 0

            try:
                with pdfplumber.open(temp_file.name) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        # Extract text from page
                        page_text = page.extract_text() or ""
                        text += page_text

                        # Extract tables from page
                        tables = page.extract_tables()
                        if tables:
                            for table_num, table in enumerate(tables, 1):
                                total_tables += 1
                                text += f"\n\n[TABLE {total_tables} - Page {page_num}]\n"
                                text += self._format_table_as_markdown(table)
                                text += "\n[END TABLE]\n\n"

                        # Add page break marker
                        text += f"\n\n--- PAGE {page_num} ---\n\n"

                    total_chars = len(text)

                    # Display extraction diagnostics
                    st.success(f"✅ **PDF Extraction Complete**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Pages", len(pdf.pages))
                    with col2:
                        st.metric("Tables Extracted", total_tables)
                    with col3:
                        st.metric("Characters", f"{total_chars:,}")

                    if total_tables == 0:
                        st.warning("⚠️ No tables detected. If this paper contains tables, "
                                 "consider using LaTeX source instead for better extraction.")

            except Exception as e:
                st.error(f"Error extracting text from PDF: {e}")

            # Clean up
            os.unlink(temp_file.name)
            return text

    def _format_table_as_markdown(self, table):
        """
        Format an extracted table as markdown.

        Args:
            table: List of lists representing table rows

        Returns:
            str: Markdown-formatted table
        """
        if not table or not any(table):
            return "[Empty table]"

        markdown = ""

        # Process table rows
        for i, row in enumerate(table):
            if row is None:
                continue

            # Clean cells (handle None values)
            cells = [str(cell).strip() if cell is not None else "" for cell in row]

            # Create markdown row
            markdown += "| " + " | ".join(cells) + " |\n"

            # Add header separator after first row
            if i == 0:
                markdown += "|" + "|".join(["---" for _ in cells]) + "|\n"

        return markdown

    def create_excel_export(self, results: Dict) -> BytesIO:
        """
        Create an Excel file with experiment tracking data.

        Returns:
            BytesIO object containing the Excel file
        """
        # Extract key data
        metadata = results.get('metadata', {})
        round_0 = results.get('round_0', {})
        active_personas = round_0.get('selected_personas', [])
        weights = round_0.get('weights', {})
        consensus = results.get('consensus', {})

        # Extract verdicts from each round
        round_1_verdicts = {}
        for persona in active_personas:
            verdict = extract_verdict(results.get('round_1', {}).get(persona, ''))
            round_1_verdicts[persona] = verdict

        round_2c_verdicts = {}
        for persona in active_personas:
            verdict = extract_verdict(results.get('round_2c', {}).get(persona, ''))
            round_2c_verdicts[persona] = verdict

        # Create configuration sheet
        config_data = {
            'Configuration Item': [],
            'Value': []
        }

        # Basic info
        config_data['Configuration Item'].extend([
            'Run Date', 'Run Time', 'Runtime Duration',
            'Model Version', 'Temperature',
            'Thinking Enabled', 'Thinking Budget (tokens)',
            'Max Retries', 'Retry Delay (seconds)'
        ])
        config_data['Value'].extend([
            metadata.get('start_time', 'N/A').split(' ')[0],
            metadata.get('start_time', 'N/A').split(' ')[1] if ' ' in metadata.get('start_time', 'N/A') else 'N/A',
            metadata.get('total_runtime_formatted', 'N/A'),
            metadata.get('model_version', 'N/A'),
            str(metadata.get('temperature', 'N/A')),
            str(metadata.get('thinking_enabled', 'N/A')),
            str(metadata.get('thinking_budget_tokens', 'N/A')),
            str(metadata.get('max_retries', 'N/A')),
            str(metadata.get('retry_delay_seconds', 'N/A'))
        ])

        # Add prompt versions
        prompt_versions = metadata.get('prompt_versions', {})
        for key, version in sorted(prompt_versions.items()):
            config_data['Configuration Item'].append(f'Prompt: {key}')
            config_data['Value'].append(version)

        config_df = pd.DataFrame(config_data)

        # Create results sheet
        results_data = {
            'Persona': [],
            'Weight': [],
            'Round 1 Verdict': [],
            'Round 2C Verdict (Final)': [],
            'Verdict Changed': []
        }

        for persona in active_personas:
            results_data['Persona'].append(persona)
            results_data['Weight'].append(weights.get(persona, 0))
            results_data['Round 1 Verdict'].append(round_1_verdicts.get(persona, 'UNKNOWN'))
            results_data['Round 2C Verdict (Final)'].append(round_2c_verdicts.get(persona, 'UNKNOWN'))
            changed = "Yes" if round_1_verdicts.get(persona) != round_2c_verdicts.get(persona) else "No"
            results_data['Verdict Changed'].append(changed)

        results_df = pd.DataFrame(results_data)

        # Create consensus sheet
        consensus_data = {
            'Metric': [
                'Weighted Consensus Score',
                'Final Decision',
                'Justification'
            ],
            'Value': [
                f"{consensus.get('weighted_score', 0):.3f}",
                consensus.get('decision', 'UNKNOWN'),
                round_0.get('justification', 'N/A')
            ]
        }
        consensus_df = pd.DataFrame(consensus_data)

        # Create cost tracking sheet
        token_usage = metadata.get('token_usage', {})
        cost_data = {
            'Metric': [],
            'Value': []
        }

        if token_usage:
            cost_usd = token_usage.get('cost_usd', {})
            llm_calls = token_usage.get('llm_calls', {})
            input_tokens = token_usage.get('input_tokens', {})
            output_tokens = token_usage.get('output_tokens', {})
            pricing = token_usage.get('pricing', {})

            cost_data['Metric'].extend([
                'Paper Tokens',
                'Input Tokens (Debate)',
                'Input Tokens (Summarization)',
                'Total Input Tokens',
                'Output Tokens (Debate)',
                'Output Tokens (Summarization)',
                'Total Output Tokens',
                'Total Tokens',
                '',
                'Input Cost (USD)',
                'Output Cost (USD)',
                'Total Cost (USD)',
                '',
                'LLM Calls (Debate)',
                'LLM Calls (Summarization)',
                'Total LLM Calls',
                '',
                'Model',
                'Input Price (per 1M tokens)',
                'Output Price (per 1M tokens)'
            ])

            cost_data['Value'].extend([
                token_usage.get('paper_tokens', 0),
                input_tokens.get('debate', 0),
                input_tokens.get('summarization', 0),
                input_tokens.get('total', 0),
                output_tokens.get('debate', 0),
                output_tokens.get('summarization', 0),
                output_tokens.get('total', 0),
                token_usage.get('total_tokens', 0),
                '',
                f"${cost_usd.get('input', 0):.4f}",
                f"${cost_usd.get('output', 0):.4f}",
                f"${cost_usd.get('total', 0):.4f}",
                '',
                llm_calls.get('debate', 0),
                llm_calls.get('summarization', 0),
                llm_calls.get('total', 0),
                '',
                pricing.get('model', 'N/A'),
                f"${pricing.get('input_per_million', 0):.2f}",
                f"${pricing.get('output_per_million', 0):.2f}"
            ])

        cost_df = pd.DataFrame(cost_data)

        # Create quote validation sheet (if available)
        quote_validation_df = None
        quote_val_meta = metadata.get('quote_validation', {})
        if quote_val_meta.get('enabled'):
            quote_data = {
                'Round': [],
                'Persona': [],
                'Quote Text': [],
                'Is Valid': [],
                'Similarity Score': [],
                'Is Mathematical': [],
                'Threshold Used': []
            }

            # Process Round 1 validation
            r1_validation = results.get('round_1_quote_validation', {})
            for persona, val_result in r1_validation.items():
                for quote in val_result.get('quotes', []):
                    quote_data['Round'].append('Round 1')
                    quote_data['Persona'].append(persona)
                    quote_data['Quote Text'].append(quote['text'][:200] + ('...' if len(quote['text']) > 200 else ''))
                    quote_data['Is Valid'].append('Yes' if quote['is_valid'] else 'No')
                    quote_data['Similarity Score'].append(f"{quote['similarity_score']:.1f}%")
                    quote_data['Is Mathematical'].append('Yes' if quote['is_mathematical'] else 'No')
                    quote_data['Threshold Used'].append(f"{quote['threshold_used']}%")

            # Process Round 2C validation
            r2c_validation = results.get('round_2c_quote_validation', {})
            for persona, val_result in r2c_validation.items():
                for quote in val_result.get('quotes', []):
                    quote_data['Round'].append('Round 2C')
                    quote_data['Persona'].append(persona)
                    quote_data['Quote Text'].append(quote['text'][:200] + ('...' if len(quote['text']) > 200 else ''))
                    quote_data['Is Valid'].append('Yes' if quote['is_valid'] else 'No')
                    quote_data['Similarity Score'].append(f"{quote['similarity_score']:.1f}%")
                    quote_data['Is Mathematical'].append('Yes' if quote['is_mathematical'] else 'No')
                    quote_data['Threshold Used'].append(f"{quote['threshold_used']}%")

            if quote_data['Round']:  # Only create if we have data
                quote_validation_df = pd.DataFrame(quote_data)

        # Write to Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            config_df.to_excel(writer, sheet_name='Configuration', index=False)
            results_df.to_excel(writer, sheet_name='Results', index=False)
            consensus_df.to_excel(writer, sheet_name='Consensus', index=False)
            if not cost_data['Metric']:  # Only add if we have cost data
                cost_df.to_excel(writer, sheet_name='Cost Tracking', index=False)
            else:
                cost_df.to_excel(writer, sheet_name='Cost Tracking', index=False)

            # Add quote validation sheet if available
            if quote_validation_df is not None:
                quote_validation_df.to_excel(writer, sheet_name='Quote Validation', index=False)

            # Auto-adjust column widths
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

        output.seek(0)
        return output

    def display_debate_results(self, results: Dict, summaries: Dict):
        """Display the multi-agent debate results with optional summarized outputs."""
        st.success("✅ Multi-Agent Evaluation Complete!")

        # Check if summarization was used
        use_summarizer = st.session_state.get('use_summarizer', False)
        has_summaries = summaries.get('round_1_summaries') and len(summaries.get('round_1_summaries', {})) > 0

        # Display cost estimate prominently
        metadata = results.get('metadata', {})
        token_usage = metadata.get('token_usage', {})
        cache_info = metadata.get('cache', {})

        if token_usage:
            cost_data = token_usage.get('cost_usd', {})
            llm_calls = token_usage.get('llm_calls', {})
            total_cost = cost_data.get('total', 0)
            total_calls = llm_calls.get('total', 0)

            mode_text = "with summarization" if has_summaries else "full output only"

            # Show cost with cache savings if applicable
            if cache_info.get('enabled') and cache_info.get('cached_rounds', 0) > 0:
                cached_rounds = cache_info.get('cached_rounds', 0)
                total_rounds = cache_info.get('total_rounds', 6)
                savings = cache_info.get('estimated_savings_usd', 0)
                cache_hit_rate = cache_info.get('cache_hit_rate', 0)

                st.success(
                    f"💰 **Estimated Cost:** ${total_cost:.4f} USD ({total_calls} LLM calls, "
                    f"{token_usage.get('total_tokens', 0):,} total tokens) - {mode_text}\n\n"
                    f"💾 **Cache:** {cached_rounds}/{total_rounds} rounds cached ({cache_hit_rate*100:.0f}% hit rate) — "
                    f"Saved ~${savings:.4f} USD"
                )
            else:
                st.info(f"💰 **Estimated Cost:** ${total_cost:.4f} USD ({total_calls} LLM calls, "
                       f"{token_usage.get('total_tokens', 0):,} total tokens) - {mode_text}")

        # Display cache details in expander
        if cache_info.get('enabled'):
            with st.expander("💾 Cache Details"):
                cache_hits = cache_info.get('cache_hits', {})

                st.markdown("**Cache Status by Round:**")
                for round_name, is_hit in cache_hits.items():
                    emoji = "✅" if is_hit else "❌"
                    status = "HIT (cached)" if is_hit else "MISS (computed)"
                    st.markdown(f"- {emoji} **{round_name}**: {status}")

                if cache_info.get('cache_key'):
                    st.markdown(f"\n**Cache Key:** `{cache_info['cache_key'][:32]}...`")
                    st.caption("This key uniquely identifies the paper, personas, weights, and model configuration.")

        # Icon mapping for all personas
        icon_map = {
            "Theorist": "🔢",
            "Empiricist": "📊",
            "Mathematician": "🔢",
            "Historian": "📚",
            "Visionary": "🚀",
            "Policymaker": "🏛️"
        }

        # Get active personas from round 0
        active_personas = results.get('round_0', {}).get('selected_personas', ["Mathematician", "Historian", "Visionary"])
        weights = results.get('round_0', {}).get('weights', {})

        # ROUND 0: Persona Selection
        if 'round_0' in results:
            st.markdown('<div class="round-header">🎯 ROUND 0: PERSONA SELECTION</div>', unsafe_allow_html=True)
            with st.expander("📋 **Selected Review Panel**", expanded=True):
                st.markdown(f"**Selected Personas:** {', '.join(active_personas)}")
                st.markdown("**Weights:**")
                for persona, weight in weights.items():
                    st.markdown(f"- **{persona}**: {weight}")
                if 'justification' in results['round_0']:
                    st.markdown(f"**Justification:** {results['round_0']['justification']}")

            # System Prompt Display
            with st.expander("🔍 **View System Prompt for Round 0**", expanded=False):
                st.code(SELECTION_PROMPT, language="text")

            st.markdown("---")

        # ROUND 1: Independent Evaluation
        st.markdown('<div class="round-header">⚡ ROUND 1: INDEPENDENT EVALUATION</div>', unsafe_allow_html=True)
        st.markdown("*Each persona independently evaluates the paper with severity-weighted findings.*")

        if has_summaries:
            st.markdown('<span class="summary-badge">SUMMARIZED VIEW</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="summary-badge" style="background: #28a745;">FULL OUTPUT</span>', unsafe_allow_html=True)

        # System Prompt Display for Round 1
        with st.expander("🔍 **View System Prompts for Round 1**", expanded=False):
            for persona in active_personas:
                st.markdown(f"**{persona} System Prompt:**")
                st.code(SYSTEM_PROMPTS.get(persona, "N/A"), language="text")
                st.markdown("---")

        for role in active_personas:
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"

            if has_summaries:
                # Display SUMMARIZED version with full in expander
                summary_data = summaries['round_1_summaries'].get(role, {})
                summary_text = summary_data.get('summary', 'No summary available')
                verdict = summary_data.get('verdict', 'UNKNOWN')

                with st.container():
                    st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)
                    st.markdown(f"### {icon} {role.upper()}")
                    st.markdown(f"**Verdict:** {format_verdict(verdict)}", unsafe_allow_html=True)
                    st.markdown(summary_text, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Full report in expander
                    with st.expander(f"📄 View Full {role} Report", expanded=False):
                        raw_text = results['round_1'][role]
                        formatted_text = format_severity_labels(raw_text)
                        st.markdown(formatted_text, unsafe_allow_html=True)
            else:
                # Display FULL output directly
                raw_text = results['round_1'][role]
                formatted_text = format_severity_labels(raw_text)
                verdict = extract_verdict(raw_text)

                with st.expander(f"📄 {icon} {role.upper()} - Full Report", expanded=True):
                    st.markdown(f"**Verdict:** {format_verdict(verdict)}", unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown(formatted_text, unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 2A: Cross-Examination
        st.markdown('<div class="round-header">🔄 ROUND 2A: CROSS-EXAMINATION</div>', unsafe_allow_html=True)
        st.markdown("*Personas challenge each other's findings and ask clarifying questions.*")

        if has_summaries:
            st.markdown('<span class="summary-badge">SUMMARIZED VIEW</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="summary-badge" style="background: #28a745;">FULL OUTPUT</span>', unsafe_allow_html=True)

        # System Prompt Display for Round 2A
        with st.expander("🔍 **View System Prompt for Round 2A**", expanded=False):
            st.code(DEBATE_PROMPTS["Round_2A_Cross_Examination"], language="text")

        for role in active_personas:
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"

            if has_summaries:
                # Display SUMMARIZED version
                summary_text = summaries['round_2a_summaries'].get(role, 'No summary available')

                with st.container():
                    st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)
                    st.markdown(f"### {icon} {role.upper()} - Cross-Examination")
                    st.markdown(summary_text, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Full report in expander
                    with st.expander(f"📄 View Full {role} Cross-Examination", expanded=False):
                        raw_text = results['round_2a'][role]
                        formatted_text = format_severity_labels(raw_text)

                        # Parse and display structured sections
                        insights_pattern = r'\*\*Cross-Domain Insights[:\*]*\s*(.*?)(?=\*\*Constructive Pushback|\*\*Clarification Requests|\Z)'
                        insights_match = re.search(insights_pattern, formatted_text, re.DOTALL | re.IGNORECASE)

                        if insights_match:
                            st.markdown("**Cross-Domain Insights:**")
                            insights_text = insights_match.group(1).strip()
                            st.markdown(insights_text, unsafe_allow_html=True)
                            st.markdown("")

                        pushback_pattern = r'\*\*Constructive Pushback[:\*]*\s*(.*?)(?=\*\*Clarification Requests|\Z)'
                        pushback_match = re.search(pushback_pattern, formatted_text, re.DOTALL | re.IGNORECASE)

                        if pushback_match:
                            st.markdown("**Constructive Pushback:**")
                            pushback_text = pushback_match.group(1).strip()
                            st.markdown(pushback_text, unsafe_allow_html=True)
                            st.markdown("")

                        clarification_pattern = r'\*\*Clarification Requests[:\*]*\s*(.*?)(?=\Z)'
                        clarification_match = re.search(clarification_pattern, formatted_text, re.DOTALL | re.IGNORECASE)

                        if clarification_match:
                            st.markdown("**Clarification Requests:**")
                            clarification_text = clarification_match.group(1).strip()
                            st.markdown(clarification_text, unsafe_allow_html=True)

                        if not (insights_match or pushback_match or clarification_match):
                            st.markdown(formatted_text, unsafe_allow_html=True)
            else:
                # Display FULL output directly
                raw_text = results['round_2a'][role]
                formatted_text = format_severity_labels(raw_text)

                with st.expander(f"📄 {icon} {role.upper()} - Cross-Examination (Full)", expanded=True):
                    st.markdown(formatted_text, unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 2B: Direct Examination
        st.markdown('<div class="round-header">💬 ROUND 2B: ANSWERING QUESTIONS</div>', unsafe_allow_html=True)
        st.markdown("*Personas respond to peer questions with evidence and concessions/defenses.*")

        if has_summaries:
            st.markdown('<span class="summary-badge">SUMMARIZED VIEW</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="summary-badge" style="background: #28a745;">FULL OUTPUT</span>', unsafe_allow_html=True)

        # System Prompt Display for Round 2B
        with st.expander("🔍 **View System Prompt for Round 2B**", expanded=False):
            st.code(DEBATE_PROMPTS["Round_2B_Direct_Examination"], language="text")

        for role in active_personas:
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"

            if has_summaries:
                # Display SUMMARIZED version
                summary_text = summaries['round_2b_summaries'].get(role, 'No summary available')

                with st.container():
                    st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)
                    st.markdown(f"### {icon} {role.upper()} - Responses")
                    st.markdown(summary_text, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Full report in expander
                    with st.expander(f"📄 View Full {role} Responses", expanded=False):
                        raw_text = results['round_2b'][role]
                        peers = [p for p in active_personas if p != role]

                        for peer in peers:
                            pattern = rf'\*\*Response to {peer}[:\*]*\s*(.*?)(?=\*\*Response to |\*\*Concession or Defense|\Z)'
                            match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)

                            if match:
                                response_content = match.group(1).strip()
                                st.markdown(f"**Response to {peer}:**")
                                st.markdown(response_content)
                                st.markdown("---")

                        # General concession or defense section
                        pattern = r'\*\*Concession or Defense[:\*]*\s*(.*?)(?=\Z)'
                        match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
                        if match:
                            content = match.group(1).strip()
                            st.markdown(f"**Overall Position:**")
                            st.markdown(content)
            else:
                # Display FULL output directly
                raw_text = results['round_2b'][role]
                formatted_text = format_severity_labels(raw_text)

                with st.expander(f"📄 {icon} {role.upper()} - Responses (Full)", expanded=True):
                    st.markdown(formatted_text, unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 2C: Final Amendments
        st.markdown('<div class="round-header">⚖️ ROUND 2C: FINAL AMENDMENTS</div>', unsafe_allow_html=True)
        st.markdown("*Personas submit final verdicts after integrating full debate.*")

        if has_summaries:
            st.markdown('<span class="summary-badge">SUMMARIZED VIEW</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="summary-badge" style="background: #28a745;">FULL OUTPUT</span>', unsafe_allow_html=True)

        # System Prompt Display for Round 2C
        with st.expander("🔍 **View System Prompt for Round 2C**", expanded=False):
            st.code(DEBATE_PROMPTS["Round_2C_Final_Amendment"], language="text")

        if has_summaries:
            cols = st.columns(3)
            for idx, role in enumerate(active_personas):
                icon = icon_map.get(role, "🔍")
                box_class = f"{role.lower()}-box"

                # Display SUMMARIZED version
                summary_data = summaries['round_2c_summaries'].get(role, {})
                summary_text = summary_data.get('summary', 'No summary available')
                verdict = summary_data.get('verdict', 'UNKNOWN')

                with cols[idx]:
                    st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)
                    st.markdown(f"### {icon} {role.upper()}")
                    st.markdown(f"**Final Verdict:** {format_verdict(verdict)}", unsafe_allow_html=True)
                    st.markdown(summary_text, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Full report in expander
                    with st.expander(f"📄 Full Report", expanded=False):
                        raw_text = results['round_2c'][role]
                        formatted_text = format_severity_labels(raw_text)
                        st.markdown(formatted_text, unsafe_allow_html=True)
        else:
            # Display FULL output directly
            for role in active_personas:
                icon = icon_map.get(role, "🔍")
                raw_text = results['round_2c'][role]
                formatted_text = format_severity_labels(raw_text)
                verdict = extract_verdict(raw_text)

                with st.expander(f"📄 {icon} {role.upper()} - Final Amendment (Full)", expanded=True):
                    st.markdown(f"**Final Verdict:** {format_verdict(verdict)}", unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown(formatted_text, unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 3: Consensus Calculation & Editor Decision (SUMMARIZED)
        st.markdown('<div class="round-header">📜 ROUND 3: WEIGHTED CONSENSUS & EDITOR DECISION</div>', unsafe_allow_html=True)

        # System Prompt Display for Round 3
        with st.expander("🔍 **View System Prompt for Round 3**", expanded=False):
            st.code(DEBATE_PROMPTS["Round_3_Editor"], language="text")

        # Display deterministic consensus calculation
        if 'consensus' in results:
            st.markdown("### 🔢 Deterministic Weighted Consensus")
            consensus = results['consensus']

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("**Individual Verdicts & Weights:**")
                for persona in active_personas:
                    verdict = consensus['verdicts'].get(persona, 'UNKNOWN')
                    weight = weights.get(persona, 0)
                    verdict_value = 1.0 if verdict == 'PASS' else 0.5 if verdict == 'REVISE' else 0.0
                    contribution = weight * verdict_value
                    st.markdown(f"- **{persona}** (weight={weight}): {verdict} = {verdict_value} → contribution = {contribution:.3f}")

            with col2:
                st.markdown("**Consensus Score:**")
                st.metric("Weighted Score", f"{consensus['weighted_score']:.3f}", help="Out of 1.0")
                st.markdown("**Decision Threshold:**")
                st.code("""Score > 0.75: ACCEPT
0.40 ≤ Score ≤ 0.75: REJECT & RESUBMIT
Score < 0.40: REJECT""", language="text")

            st.markdown(f"""
            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; border: 2px solid #388e3c; margin: 15px 0;">
                <strong style="font-size: 18px;">📊 Computed Decision: {consensus['decision']}</strong>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

        # Editor Report
        if has_summaries:
            st.markdown('<span class="summary-badge">SUMMARIZED VIEW</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="summary-badge" style="background: #28a745;">FULL OUTPUT</span>', unsafe_allow_html=True)

        final_decision = results.get('consensus', {}).get('decision', 'UNKNOWN')

        if has_summaries:
            editor_summary = summaries.get('editor_summary', 'No summary available')

            st.markdown(f"""
            <div class="persona-box editor-box">
            <h3>🏛️ Senior Editor's Report</h3>
            <div style="text-align: center; margin: 20px 0;">
            {format_final_verdict(final_decision)}
            </div>
            <p style="text-align: center; color: #666; font-size: 14px;">
            (Decision computed deterministically from weighted consensus)
            </p>
            """, unsafe_allow_html=True)

            st.markdown(editor_summary, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Display full editor report
            with st.expander("📄 🏛️ Senior Editor's Full Report", expanded=True):
                st.markdown(f"**Decision:** {format_final_verdict(final_decision)}", unsafe_allow_html=True)
                st.markdown("---")
                editor_text = results.get('final_decision', 'No editor report available')
                formatted_editor = format_severity_labels(editor_text)
                st.markdown(formatted_editor, unsafe_allow_html=True)

        # Full editor report in expander
        with st.expander("📄 View Full Editor Report", expanded=False):
            final_text = results['final_decision']
            sections = re.split(r'\*\*([^*]+)\*\*', final_text)
            for i in range(0, len(sections), 2):
                if i + 1 < len(sections):
                    header = sections[i + 1].strip().rstrip(':')
                    content = sections[i + 2].strip() if i + 2 < len(sections) else ""
                    st.markdown(f"**{header}:**")
                    st.markdown(content, unsafe_allow_html=True)

        st.markdown("---")

        # Summary Table
        st.subheader("📊 Evaluation Summary")

        # Create table data
        table_data = []
        r1_verdicts = {p: extract_verdict(results['round_1'].get(p, '')) for p in active_personas}
        r2c_verdicts = {p: extract_verdict(results['round_2c'].get(p, '')) for p in active_personas}

        for persona in active_personas:
            weight = weights.get(persona, 0)
            r1_v = r1_verdicts[persona]
            r2c_v = r2c_verdicts[persona]
            changed = "Yes ✅" if r1_v != r2c_v else "No ➖"
            table_data.append({
                "Persona": persona,
                "Weight": weight,
                "Round 1 Verdict": r1_v,
                "Final Verdict (R2C)": r2c_v,
                "Changed?": changed
            })

        # Display as dataframe
        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        weighted_score = results.get('consensus', {}).get('weighted_score', 0)

        st.markdown(f"""
        **FINAL EDITORIAL DECISION (Weighted Score: {weighted_score:.3f}):** {format_final_verdict(final_decision)}
        """, unsafe_allow_html=True)

        # Metadata Section
        if 'metadata' in results:
            st.markdown("---")
            st.subheader("⚙️ Execution Metadata")
            metadata = results['metadata']

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Model Configuration:**")
                thinking_status = "Enabled" if metadata.get('thinking_enabled') else "Disabled"
                thinking_budget = f"{metadata.get('thinking_budget_tokens', 'N/A')} tokens" if metadata.get('thinking_enabled') else "N/A"

                if 'max_tokens_round_2c' in metadata:
                    max_tokens_display = f"""Max Output Tokens:
  - Rounds 0,1,2A,2B: {metadata.get('max_tokens_round_0_1_2a_2b', 'N/A')}
  - Round 2C: {metadata.get('max_tokens_round_2c', 'N/A')}
  - Round 3 (Editor): {metadata.get('max_tokens_round_3_editor', 'N/A')}"""
                else:
                    max_tokens_display = f"Max Output Tokens: {metadata.get('max_tokens', 'N/A')}"

                st.code(f"""Model: {metadata.get('model_version', 'N/A')}
Temperature: {metadata.get('temperature', 'N/A')}
{max_tokens_display}
Thinking Mode: {thinking_status}
Thinking Budget: {thinking_budget}
Max Retries: {metadata.get('max_retries', 'N/A')}
Retry Delay: {metadata.get('retry_delay_seconds', 'N/A')}s""", language="text")

            with col2:
                st.markdown("**Execution Time:**")
                st.code(f"""Start Time: {metadata.get('start_time', 'N/A')}
End Time: {metadata.get('end_time', 'N/A')}
Total Runtime: {metadata.get('total_runtime_formatted', 'N/A')}
({metadata.get('total_runtime_seconds', 'N/A')} seconds)""", language="text")

            # Quote Validation Results
            quote_val_meta = metadata.get('quote_validation', {})
            if quote_val_meta.get('enabled'):
                st.markdown("---")
                st.subheader("🔍 Quote Validation")

                col_q1, col_q2 = st.columns(2)

                with col_q1:
                    st.markdown("**Round 1 Validation:**")
                    r1_summary = quote_val_meta.get('round_1', {})
                    if r1_summary and 'error' not in r1_summary:
                        total = r1_summary.get('total_quotes_found', 0)
                        valid = r1_summary.get('valid_quotes', 0)
                        invalid = r1_summary.get('invalid_quotes', 0)
                        rate = r1_summary.get('validation_rate', 0)

                        st.code(f"""Total Quotes: {total}
Valid: {valid}
Invalid: {invalid}
Validation Rate: {rate:.1f}%""", language="text")

                        if invalid > 0:
                            with st.expander("⚠️ View Unverified Quotes", expanded=False):
                                for detail in r1_summary.get('invalid_quote_details', []):
                                    st.markdown(f"**{detail['persona']}** (Score: {detail['similarity_score']:.1f}%)")
                                    st.text(detail['quote'])
                                    st.markdown("---")
                    else:
                        st.warning("Validation unavailable or failed")

                with col_q2:
                    st.markdown("**Round 2C Validation:**")
                    r2c_summary = quote_val_meta.get('round_2c', {})
                    if r2c_summary and 'error' not in r2c_summary:
                        total = r2c_summary.get('total_quotes_found', 0)
                        valid = r2c_summary.get('valid_quotes', 0)
                        invalid = r2c_summary.get('invalid_quotes', 0)
                        rate = r2c_summary.get('validation_rate', 0)

                        st.code(f"""Total Quotes: {total}
Valid: {valid}
Invalid: {invalid}
Validation Rate: {rate:.1f}%""", language="text")

                        if invalid > 0:
                            with st.expander("⚠️ View Unverified Quotes", expanded=False):
                                for detail in r2c_summary.get('invalid_quote_details', []):
                                    st.markdown(f"**{detail['persona']}** (Score: {detail['similarity_score']:.1f}%)")
                                    st.text(detail['quote'])
                                    st.markdown("---")
                    else:
                        st.warning("Validation unavailable or failed")

                # Note about fuzzy matching
                if not r1_summary.get('fuzzy_matching_available', False):
                    st.info("ℹ️ **Note:** Fuzzy matching not available. Install thefuzz for improved validation: `pip install thefuzz python-Levenshtein`")

            # Cost and Token Usage
            if 'token_usage' in metadata:
                st.markdown("---")
                st.subheader("💰 Cost & Token Usage")

                token_data = metadata['token_usage']
                cost_data = token_data.get('cost_usd', {})
                llm_calls = token_data.get('llm_calls', {})

                col3, col4 = st.columns(2)

                with col3:
                    st.markdown("**Token Breakdown:**")
                    st.code(f"""Paper Tokens: {token_data.get('paper_tokens', 0):,}
Input Tokens (Debate): {token_data.get('input_tokens', {}).get('debate', 0):,}
Input Tokens (Summarization): {token_data.get('input_tokens', {}).get('summarization', 0):,}
Total Input: {token_data.get('input_tokens', {}).get('total', 0):,}

Output Tokens (Debate): {token_data.get('output_tokens', {}).get('debate', 0):,}
Output Tokens (Summarization): {token_data.get('output_tokens', {}).get('summarization', 0):,}
Total Output: {token_data.get('output_tokens', {}).get('total', 0):,}

TOTAL TOKENS: {token_data.get('total_tokens', 0):,}""", language="text")

                with col4:
                    st.markdown("**Cost Breakdown:**")
                    pricing = token_data.get('pricing', {})
                    st.code(f"""Model: {pricing.get('model', 'N/A')}
Pricing:
  Input: ${pricing.get('input_per_million', 0)}/1M tokens
  Output: ${pricing.get('output_per_million', 0)}/1M tokens

Input Cost: ${cost_data.get('input', 0):.4f}
Output Cost: ${cost_data.get('output', 0):.4f}
TOTAL COST: ${cost_data.get('total', 0):.4f}

LLM Calls:
  Debate: {llm_calls.get('debate', 0)}
  Summarization: {llm_calls.get('summarization', 0)}
  Total: {llm_calls.get('total', 0)}""", language="text")

        # Download options
        st.markdown("---")
        st.subheader("📥 Download Reports")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Prepare full report (with summaries)
            full_report = f"""# Multi-Agent Referee Evaluation Report (Summarized Version)
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

NOTE: This report includes both summarized and full outputs from the multi-agent debate.

## ROUND 0: PERSONA SELECTION

Selected Personas: {', '.join(active_personas)}
Weights: {json.dumps(weights, indent=2)}
Justification: {results.get('round_0', {}).get('justification', 'N/A')}

---

## ROUND 1: INDEPENDENT EVALUATION

"""
            for role in active_personas:
                summary_data = summaries['round_1_summaries'].get(role, {})
                full_report += f"### {role} (Summary)\n{summary_data.get('summary', 'N/A')}\n\n"
                full_report += f"### {role} (Full Report)\n{results['round_1'][role]}\n\n"

            full_report += """---

## ROUND 2A: CROSS-EXAMINATION

"""
            for role in active_personas:
                summary_text = summaries['round_2a_summaries'].get(role, 'N/A')
                full_report += f"### {role} (Summary)\n{summary_text}\n\n"
                full_report += f"### {role} (Full Report)\n{results['round_2a'][role]}\n\n"

            full_report += """---

## ROUND 2B: ANSWERING QUESTIONS

"""
            for role in active_personas:
                summary_text = summaries['round_2b_summaries'].get(role, 'N/A')
                full_report += f"### {role} (Summary)\n{summary_text}\n\n"
                full_report += f"### {role} (Full Report)\n{results['round_2b'][role]}\n\n"

            full_report += """---

## ROUND 2C: FINAL AMENDMENTS

"""
            for role in active_personas:
                summary_data = summaries['round_2c_summaries'].get(role, {})
                full_report += f"### {role} (Summary)\n{summary_data.get('summary', 'N/A')}\n\n"
                full_report += f"### {role} (Full Report)\n{results['round_2c'][role]}\n\n"

            # Add consensus calculation
            if 'consensus' in results:
                consensus = results['consensus']
                full_report += f"""---

## ROUND 3: WEIGHTED CONSENSUS CALCULATION

### Deterministic Consensus Score: {consensus['weighted_score']:.3f}

**Individual Verdicts:**
"""
                for persona in active_personas:
                    verdict = consensus['verdicts'].get(persona, 'UNKNOWN')
                    weight = weights.get(persona, 0)
                    verdict_value = 1.0 if verdict == 'PASS' else 0.5 if verdict == 'REVISE' else 0.0
                    contribution = weight * verdict_value
                    full_report += f"- {persona} (weight={weight}): {verdict} = {verdict_value} → contribution = {contribution:.3f}\n"

                full_report += f"""
**Decision Thresholds:**
- Score > 0.75: ACCEPT
- 0.40 ≤ Score ≤ 0.75: REJECT AND RESUBMIT
- Score < 0.40: REJECT

**Computed Decision:** {consensus['decision']}

---

## EDITOR'S REPORT

### Summary
{summaries.get('editor_summary', 'N/A')}

### Full Report
{results['final_decision']}
"""

            st.download_button(
                label="📄 Download Full Evaluation Report",
                data=full_report,
                file_name=f"multi_agent_evaluation_summarized_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

        with col2:
            # Prepare summary table in markdown
            summary_md = f"""# Evaluation Summary Table
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

| Persona | Weight | Round 1 Verdict | Final Verdict (R2C) | Changed? |
|---------|--------|----------------|---------------------|----------|
"""
            for persona in active_personas:
                weight = weights.get(persona, 0)
                r1_v = r1_verdicts[persona]
                r2c_v = r2c_verdicts[persona]
                changed = "Yes" if r1_v != r2c_v else "No"
                summary_md += f"| {persona} | {weight} | {r1_v} | {r2c_v} | {changed} |\n"

            summary_md += f"\n**FINAL EDITORIAL DECISION:** {final_decision}\n"

            st.download_button(
                label="📊 Download Summary Table",
                data=summary_md,
                file_name=f"evaluation_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

        with col3:
            # Create Excel export for experiment tracking
            excel_data = self.create_excel_export(results)

            st.download_button(
                label="📊 Download Excel (Experiment Tracking)",
                data=excel_data,
                file_name=f"experiment_tracking_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download structured Excel file with configurations, prompt versions, and verdicts for experiment comparison"
            )

        # --- Add comprehensive ZIP download ---
        st.markdown("---")
        st.markdown("### 📦 Download All Files")

        # Extract paper name from session state manuscript_file (remove extension)
        manuscript_file = st.session_state.get('manuscript_file', 'paper')
        paper_name = manuscript_file.rsplit('.', 1)[0] if '.' in manuscript_file else manuscript_file
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create ZIP file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add full evaluation report
            zip_file.writestr(
                f"{paper_name}_{timestamp}_full_report.md",
                full_report
            )
            # Add summary table
            zip_file.writestr(
                f"{paper_name}_{timestamp}_summary.md",
                summary_md
            )
            # Add Excel file (excel_data is BytesIO, need to get bytes)
            zip_file.writestr(
                f"{paper_name}_{timestamp}_experiment_tracking.xlsx",
                excel_data.getvalue()  # Get bytes from BytesIO object
            )

        zip_buffer.seek(0)

        st.download_button(
            label="📦 Download All Files (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{paper_name}_{timestamp}_complete_evaluation.zip",
            mime="application/zip",
            help="Download all 3 files in a single ZIP archive",
            type="primary"
        )
