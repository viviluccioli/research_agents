# -*- coding: utf-8 -*-
# workflow_exp_4.py - Experiment 4 Multi-Agent Referee Report UI
"""
Experimental version with 10 personas (selects 3) based on mad_experiments/exp_4.

This version uses the engine_exp_4 module which includes 10 persona options
instead of the original 5. It uses LLM-powered summarization to compress
debate outputs for cleaner display while preserving full reports in expandable sections.
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
from referee.engine_exp_4 import execute_debate_pipeline, SELECTION_PROMPT, SYSTEM_PROMPTS, DEBATE_PROMPTS
from referee._utils.summarizer import summarize_all_rounds
from referee._utils.pdf_extractor_v2 import extract_pdf_with_figures, PYMUPDF_AVAILABLE, ExtractedContent
from config import USE_PYMUPDF, PYMUPDF_MIN_FIGURE_SIZE, PYMUPDF_RESOLUTION_SCALE, PYMUPDF_EXTRACT_TABLES

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

### 👥 Available Personas (10 personas, 3 will be selected)
            """)

            # Compact horizontal persona cards (10 personas)
            st.markdown("""
            <style>
            .persona-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
                padding: 12px;
                margin: 5px;
                text-align: center;
                color: white;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                flex: 1;
                min-width: 160px;
            }
            .persona-card h4 {
                margin: 0 0 6px 0;
                font-size: 16px;
            }
            .persona-card p {
                margin: 3px 0;
                font-size: 11px;
                opacity: 0.95;
            }
            .persona-theorist { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .persona-econometrician { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
            .persona-ml-expert { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
            .persona-data-scientist { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
            .persona-cs-expert { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
            .persona-historian { background: linear-gradient(135deg, #7f7fd5 0%, #86a8e7 100%); }
            .persona-visionary { background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%); }
            .persona-policymaker { background: linear-gradient(135deg, #ee5a6f 0%, #f29263 100%); }
            .persona-ethicist { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); }
            .persona-perspective { background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); }
            </style>
            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin: 15px 0;">
                <div class="persona-card persona-theorist">
                    <h4>🔢 Theorist</h4>
                    <p><strong>Focus:</strong> Mathematical logic & proofs</p>
                </div>
                <div class="persona-card persona-econometrician">
                    <h4>📊 Econometrician</h4>
                    <p><strong>Focus:</strong> Causal inference & identification</p>
                </div>
                <div class="persona-card persona-ml-expert">
                    <h4>🤖 ML Expert</h4>
                    <p><strong>Focus:</strong> Model architecture & hyperparameters</p>
                </div>
                <div class="persona-card persona-data-scientist">
                    <h4>📈 Data Scientist</h4>
                    <p><strong>Focus:</strong> Data pipeline & preprocessing</p>
                </div>
                <div class="persona-card persona-cs-expert">
                    <h4>💻 CS Expert</h4>
                    <p><strong>Focus:</strong> Algorithms & complexity</p>
                </div>
                <div class="persona-card persona-historian">
                    <h4>📚 Historian</h4>
                    <p><strong>Focus:</strong> Literature lineage</p>
                </div>
                <div class="persona-card persona-visionary">
                    <h4>🚀 Visionary</h4>
                    <p><strong>Focus:</strong> Paradigm shifts & novelty</p>
                </div>
                <div class="persona-card persona-policymaker">
                    <h4>🏛️ Policymaker</h4>
                    <p><strong>Focus:</strong> Policy relevance & welfare</p>
                </div>
                <div class="persona-card persona-ethicist">
                    <h4>⚖️ Ethicist</h4>
                    <p><strong>Focus:</strong> Moral values & accountability</p>
                </div>
                <div class="persona-card persona-perspective">
                    <h4>🌍 Perspective</h4>
                    <p><strong>Focus:</strong> DEI & distributional impacts</p>
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
            available_personas = [
                "Theorist", "Econometrician", "ML_Expert", "Data_Scientist", "CS_Expert",
                "Historian", "Visionary", "Policymaker", "Ethicist", "Perspective"
            ]

            # Display in 2 rows of 5
            st.markdown("**Row 1:**")
            cols1 = st.columns(5)
            st.markdown("**Row 2:**")
            cols2 = st.columns(5)
            selected_personas_list = []

            for idx, persona in enumerate(available_personas):
                col = cols1[idx] if idx < 5 else cols2[idx - 5]
                with col:
                    display_name = persona.replace("_", " ")
                    if st.checkbox(display_name, key=f"persona_check_{persona}"):
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
            available_personas = [
                "Theorist", "Econometrician", "ML_Expert", "Data_Scientist", "CS_Expert",
                "Historian", "Visionary", "Policymaker", "Ethicist", "Perspective"
            ]

            st.info("💡 Weights must sum to 1.0. Higher weight = more influence on final decision")

            selected_personas_dict = {}

            # Display checkboxes in 2 rows of 5
            st.markdown("**Row 1:**")
            cols_check1 = st.columns(5)
            st.markdown("**Row 2:**")
            cols_check2 = st.columns(5)
            selected_for_weight = []

            # First row and second row: checkboxes
            for idx, persona in enumerate(available_personas):
                col = cols_check1[idx] if idx < 5 else cols_check2[idx - 5]
                with col:
                    display_name = persona.replace("_", " ")
                    if st.checkbox(display_name, key=f"persona_check_manual_{persona}"):
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
            if st.button("🚀 Step 1: Extract & Review Text", type="primary", width="stretch"):
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

                if st.button("🗑️ Clear Cache", help="Clear cached results for this specific paper", width="stretch"):
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
                    if st.button("🧹 Clear Old Cache (>30 days)", width="stretch"):
                        removed, total = clear_cache(older_than_days=30)
                        st.success(f"✅ Removed {removed}/{total} old entries")
                        st.rerun()
                with col_clear_all:
                    if st.button("⚠️ Clear All Cache", width="stretch", type="secondary"):
                        removed, total = clear_cache(older_than_days=0)
                        st.success(f"✅ Removed all {removed} entries")
                        st.rerun()

            st.markdown("---")

            col_run, col_reset = st.columns([3, 1])

            with col_run:
                run_debate = st.button("🚀 Step 2: Run Multi-Agent Evaluation", type="primary", width="stretch")

            with col_reset:
                if st.button("🔄 Start Over", width="stretch"):
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
        Extract text, tables, and figures from a PDF file.

        Uses PyMuPDF for advanced extraction (if enabled), falls back to pdfplumber.

        Returns:
            str: Extracted text with tables formatted as markdown

        Side effects:
            Stores extracted figures in st.session_state['debate_state']['figures']
        """
        try:
            # Use new PyMuPDF-based extractor
            result: ExtractedContent = extract_pdf_with_figures(
                file_content=file_content,
                use_pymupdf=USE_PYMUPDF,
                min_figure_size=PYMUPDF_MIN_FIGURE_SIZE,
                resolution_scale=PYMUPDF_RESOLUTION_SCALE,
                extract_tables=PYMUPDF_EXTRACT_TABLES
            )

            text = result.text
            total_chars = len(text)

            # Store figures in session state for future vision integration
            if 'debate_state' not in st.session_state:
                st.session_state['debate_state'] = {}
            st.session_state['debate_state']['figures'] = result.figures
            st.session_state['debate_state']['extraction_metadata'] = result.metadata

            # Display extraction diagnostics
            extractor_badge = "🚀 PyMuPDF" if result.extractor_used == "pymupdf" else "📄 pdfplumber"
            st.success(f"✅ **PDF Extraction Complete** ({extractor_badge})")

            # Show detailed metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pages", result.metadata.get('total_pages', '?'))
            with col2:
                st.metric("Figures", result.metadata.get('total_figures', 0))
            with col3:
                st.metric("Tables", result.metadata.get('total_tables', 0))
            with col4:
                st.metric("Characters", f"{total_chars:,}")

            # Additional info about extraction
            if result.metadata.get('has_multi_column'):
                st.info(f"ℹ️ Detected multi-column layout on pages: {', '.join(map(str, result.metadata.get('multi_column_pages', [])))}")

            if result.extractor_used == "pdfplumber":
                st.warning("⚠️ Using pdfplumber (fallback mode). Figure extraction not available. "
                          "To enable figure extraction, ensure PyMuPDF is installed and USE_PYMUPDF=true in .env")

            # Show figure extraction summary if available
            if result.figures:
                with st.expander(f"📊 Extracted {len(result.figures)} Figure(s)", expanded=False):
                    for fig in result.figures:
                        st.markdown(f"**{fig.figure_id}** (Page {fig.page_number})")
                        if fig.caption:
                            st.caption(fig.caption[:150] + "..." if len(fig.caption) > 150 else fig.caption)
                        st.image(BytesIO(fig.image_data), width=200)
                        st.markdown("---")

            return text

        except Exception as e:
            st.error(f"⚠️ PDF extraction error: {e}")
            st.warning("Falling back to basic pdfplumber extraction...")

            # Fallback to original pdfplumber method
            return self._extract_text_pdfplumber_fallback(file_content)

    def _extract_text_pdfplumber_fallback(self, file_content):
        """
        Fallback PDF extraction using pdfplumber (original implementation).

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
                    st.success(f"✅ **PDF Extraction Complete** (pdfplumber fallback)")
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

        # Create deduplication sheet (if available)
        deduplication_df = None
        dedup_meta = metadata.get('deduplication', {})
        if dedup_meta.get('enabled'):
            # Create summary dataframe
            dedup_summary_data = {
                'Metric': [
                    'Findings Before Deduplication',
                    'Findings After Deduplication',
                    'Clusters Merged',
                    'Reduction Rate (%)',
                    'Semantic Embeddings Available',
                    'Similarity Method'
                ],
                'Value': [
                    dedup_meta.get('total_findings_before', 0),
                    dedup_meta.get('total_findings_after', 0),
                    dedup_meta.get('clusters_merged', 0),
                    f"{dedup_meta.get('reduction_rate', 0):.1f}%",
                    'Yes' if dedup_meta.get('embeddings_available') else 'No',
                    'Multi-metric (semantic + keywords)' if dedup_meta.get('embeddings_available') else 'Keyword-based only'
                ]
            }
            deduplication_df = pd.DataFrame(dedup_summary_data)

            # Add detailed findings if available
            dedup_results = results.get('deduplication', {})
            deduplicated_findings = dedup_results.get('deduplicated_findings', [])
            if deduplicated_findings:
                findings_data = {
                    'Personas': [],
                    'Category': [],
                    'Severity': [],
                    'Is Merged': [],
                    'Source Count': [],
                    'Finding Text': []
                }

                for finding in deduplicated_findings:
                    findings_data['Personas'].append(', '.join(finding.get('personas', [])))
                    findings_data['Category'].append(finding.get('category', 'N/A'))
                    findings_data['Severity'].append(finding.get('severity', 'N/A'))
                    findings_data['Is Merged'].append('Yes' if finding.get('is_merged') else 'No')
                    findings_data['Source Count'].append(finding.get('source_count', 1))
                    findings_data['Finding Text'].append(finding.get('text', '')[:500] + ('...' if len(finding.get('text', '')) > 500 else ''))

                # Create a second sheet for detailed findings
                deduplication_details_df = pd.DataFrame(findings_data)
            else:
                deduplication_details_df = None
        else:
            deduplication_details_df = None

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

            # Add deduplication sheets if available
            if deduplication_df is not None:
                deduplication_df.to_excel(writer, sheet_name='Deduplication', index=False)
            if deduplication_details_df is not None:
                deduplication_details_df.to_excel(writer, sheet_name='Dedup Details', index=False)

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

        # Icon mapping for all 10 Exp 4 personas
        icon_map = {
            "Theorist": "🔢",
            "Econometrician": "📊",
            "ML_Expert": "🤖",
            "Data_Scientist": "📈",
            "CS_Expert": "💻",
            "Historian": "📚",
            "Visionary": "🚀",
            "Policymaker": "🏛️",
            "Ethicist": "⚖️",
            "Perspective": "🌍"
        }

        # Get active personas from round 0
        selected_personas = results.get('round_0', {}).get('selected_personas', ["Econometrician", "ML_Expert", "Policymaker"])
        weights = results.get('round_0', {}).get('weights', {})

        # IMPORTANT: Use personas that actually have results in round_1
        # This prevents KeyError if a persona was selected but failed to complete
        actual_round_1_personas = list(results.get('round_1', {}).keys())
        active_personas = actual_round_1_personas if actual_round_1_personas else selected_personas

        # ROUND 0: Persona Selection
        if 'round_0' in results:
            st.markdown('<div class="round-header">🎯 ROUND 0: PERSONA SELECTION</div>', unsafe_allow_html=True)
            with st.expander("📋 **Selected Review Panel**", expanded=True):
                st.markdown(f"**Selected Personas:** {', '.join(selected_personas)}")

                # Warn if there's a mismatch between selected and actual
                if set(selected_personas) != set(active_personas):
                    st.warning(
                        f"⚠️ **Note:** Only {len(active_personas)} of {len(selected_personas)} personas completed successfully. "
                        f"Displaying results for: {', '.join(active_personas)}"
                    )

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

        # Filter to personas that actually completed this round
        round_2a_personas = [p for p in active_personas if p in results.get('round_2a', {})]

        for role in round_2a_personas:
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

        # Filter to personas that actually completed this round
        round_2b_personas = [p for p in active_personas if p in results.get('round_2b', {})]

        for role in round_2b_personas:
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

        # Filter to personas that actually completed this round
        round_2c_personas = [p for p in active_personas if p in results.get('round_2c', {})]

        if has_summaries:
            cols = st.columns(len(round_2c_personas))
            for idx, role in enumerate(round_2c_personas):
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
            for role in round_2c_personas:
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
        st.dataframe(df, width="stretch", hide_index=True)

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

            # Deduplication Results
            dedup_meta = metadata.get('deduplication', {})
            if dedup_meta.get('enabled'):
                st.markdown("---")
                st.subheader("🔗 Cross-Reference Deduplication")

                col_d1, col_d2 = st.columns(2)

                with col_d1:
                    st.markdown("**Deduplication Statistics:**")
                    before = dedup_meta.get('total_findings_before', 0)
                    after = dedup_meta.get('total_findings_after', 0)
                    merged = dedup_meta.get('clusters_merged', 0)
                    reduction = dedup_meta.get('reduction_rate', 0)

                    st.code(f"""Findings Before: {before}
Findings After: {after}
Clusters Merged: {merged}
Reduction Rate: {reduction:.1f}%""", language="text")

                with col_d2:
                    st.markdown("**Configuration:**")
                    embeddings = "Available ✓" if dedup_meta.get('embeddings_available') else "Not Available (using keyword matching)"
                    st.code(f"""Semantic Embeddings: {embeddings}
Method: {'Multi-metric (semantic + keywords)' if dedup_meta.get('embeddings_available') else 'Keyword-based only'}""", language="text")

                # Show deduplicated findings if available
                dedup_results = results.get('deduplication', {})
                deduplicated_findings = dedup_results.get('deduplicated_findings', [])

                if deduplicated_findings:
                    merged_findings = [f for f in deduplicated_findings if f.get('is_merged', False)]
                    if merged_findings:
                        with st.expander(f"🔍 View Merged Findings ({len(merged_findings)} clusters)", expanded=False):
                            for i, finding in enumerate(merged_findings, 1):
                                personas = finding.get('personas', [])
                                st.markdown(f"**Cluster {i}** — Identified by: {', '.join(personas)}")
                                st.markdown(f"*Category:* {finding.get('category', 'N/A')} | *Severity:* {finding.get('severity', 'N/A')}")
                                if finding.get('similarity_score'):
                                    st.markdown(f"*Similarity Score:* {finding['similarity_score']:.2f}")
                                st.markdown("---")

                # Note about semantic embeddings
                if not dedup_meta.get('embeddings_available'):
                    st.info("ℹ️ **Note:** For improved semantic similarity, install sentence-transformers: `pip install sentence-transformers`")

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
                if role in results.get('round_1', {}):
                    summary_data = summaries['round_1_summaries'].get(role, {})
                    full_report += f"### {role} (Summary)\n{summary_data.get('summary', 'N/A')}\n\n"
                    full_report += f"### {role} (Full Report)\n{results['round_1'][role]}\n\n"

            full_report += """---

## ROUND 2A: CROSS-EXAMINATION

"""
            for role in active_personas:
                if role in results.get('round_2a', {}):
                    summary_text = summaries['round_2a_summaries'].get(role, 'N/A')
                    full_report += f"### {role} (Summary)\n{summary_text}\n\n"
                    full_report += f"### {role} (Full Report)\n{results['round_2a'][role]}\n\n"

            full_report += """---

## ROUND 2B: ANSWERING QUESTIONS

"""
            for role in active_personas:
                if role in results.get('round_2b', {}):
                    summary_text = summaries['round_2b_summaries'].get(role, 'N/A')
                    full_report += f"### {role} (Summary)\n{summary_text}\n\n"
                    full_report += f"### {role} (Full Report)\n{results['round_2b'][role]}\n\n"

            full_report += """---

## ROUND 2C: FINAL AMENDMENTS

"""
            for role in active_personas:
                if role in results.get('round_2c', {}):
                    summary_data = summaries['round_2c_summaries'].get(role, {})
                    full_report += f"### {role} (Summary)\n{summary_data.get('summary', 'N/A')}\n\n"
                    full_report += f"### {role} (Full Report)\n{results['round_2c'][role]}\n\n"

            # Add deduplication section
            dedup_results = results.get('deduplication', {})
            dedup_stats = dedup_results.get('statistics', {})
            if dedup_stats.get('enabled'):
                full_report += """---

## ROUND 2.5: CROSS-REFERENCE DEDUPLICATION

"""
                before = dedup_stats.get('total_findings_before', 0)
                after = dedup_stats.get('total_findings_after', 0)
                merged = dedup_stats.get('clusters_merged', 0)
                reduction = dedup_stats.get('reduction_rate', 0)
                embeddings = dedup_stats.get('embeddings_available', False)

                full_report += f"""**Deduplication Summary:**
- Findings Before: {before}
- Findings After: {after}
- Clusters Merged: {merged}
- Reduction Rate: {reduction:.1f}%
- Method: {'Multi-metric (semantic + keywords)' if embeddings else 'Keyword-based only'}

"""
                # Add merged findings details
                deduplicated_findings = dedup_results.get('deduplicated_findings', [])
                merged_findings = [f for f in deduplicated_findings if f.get('is_merged', False)]

                if merged_findings:
                    full_report += f"**Merged Finding Clusters ({len(merged_findings)}):**\n\n"
                    for i, finding in enumerate(merged_findings, 1):
                        personas = ', '.join(finding.get('personas', []))
                        category = finding.get('category', 'N/A')
                        severity = finding.get('severity', 'N/A')
                        full_report += f"#### Cluster {i}\n"
                        full_report += f"- **Identified by:** {personas}\n"
                        full_report += f"- **Category:** {category} | **Severity:** {severity}\n"
                        if finding.get('similarity_score'):
                            full_report += f"- **Similarity Score:** {finding['similarity_score']:.2f}\n"
                        full_report += f"\n{finding.get('text', '')}\n\n"

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
                mime="text/markdown",
                key="download_full_report"
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
                mime="text/markdown",
                key="download_summary_table"
            )

        with col3:
            # Create Excel export for experiment tracking
            excel_data = self.create_excel_export(results)

            st.download_button(
                label="📊 Download Excel (Experiment Tracking)",
                data=excel_data,
                file_name=f"experiment_tracking_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download structured Excel file with configurations, prompt versions, and verdicts for experiment comparison",
                key="download_excel_tracking"
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
            type="primary",
            key="download_all_zip"
        )
