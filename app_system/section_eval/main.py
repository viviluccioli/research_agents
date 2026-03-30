"""
Streamlit UI for the Section Evaluator workflow.

Entry point: instantiate SectionEvaluatorApp and call render_ui(files).
"""

import csv
import io
import streamlit as st
from typing import Dict, List, Optional, Any
from datetime import datetime
from fpdf import FPDF

from .text_extraction import decode_file
from .section_detection import (
    detect_sections, extract_sections_from_text, search_missing_section, DEFAULT_SECTIONS
)
from .hierarchy import group_subsections
from .evaluator import SectionEvaluator
from .scoring import compute_overall_score, score_bar_html
from .criteria.base import PAPER_TYPES, PAPER_TYPE_LABELS, SECTION_DEFAULTS


# ---------------------------------------------------------------------------
# App class
# ---------------------------------------------------------------------------

class SectionEvaluatorApp:
    """
    Streamlit UI for paper-type-aware section evaluation.
    Wraps SectionEvaluator and manages all session state.
    """

    CACHE_PREFIX = "se_v3"

    def __init__(self, llm=None):
        if llm is None:
            from utils import cm  # parent eval/ utils.py
            llm = cm
        self.llm = llm
        # Initialize evaluator with a shared cache stored in session state
        cache_key = f"{self.CACHE_PREFIX}_eval_cache"
        st.session_state.setdefault(cache_key, {})
        self._evaluator = SectionEvaluator(llm, cache_store=st.session_state[cache_key])

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def render_ui(self, files: Optional[Dict[str, bytes]] = None, paper_type: Optional[str] = None):
        st.subheader("Section-by-Section Manuscript Evaluation")
        st.write("Evaluate economics paper sections with paper-type-specific criteria, weighted scoring, and quote-backed assessments.")

        # --- Step 0: Paper type selection ---
        self._render_paper_type_selector()

        paper_type = st.session_state.get(f"{self.CACHE_PREFIX}_paper_type")
        if not paper_type:
            st.info("Select a paper type above to begin.")
            return

        # --- Step 1b: Figures/tables location ---
        figures_external = st.checkbox(
            "Tables and figures are in the appendix or not embedded in the uploaded text",
            key=f"{self.CACHE_PREFIX}_figures_external",
            help=(
                "Check this if your results and other sections reference tables or figures "
                "that appear in an appendix or separate file. The evaluator will not penalize "
                "sections for the absence of inline figures/tables."
            ),
        )

        # --- Step 1: Choose input mode ---
        st.write("---")
        st.write("**Step 2: Input your paper**")
        input_mode = st.radio(
            "How would you like to provide your paper?",
            ["Upload file (auto-detect sections)", "Paste sections manually"],
            key=f"{self.CACHE_PREFIX}_input_mode",
            horizontal=True,
        )

        if input_mode == "Upload file (auto-detect sections)":
            self._render_auto_detect_flow(files, paper_type, figures_external)
        else:
            self._render_freeform_flow(paper_type, figures_external)

    # ------------------------------------------------------------------
    # Paper type selector
    # ------------------------------------------------------------------

    def _render_paper_type_selector(self):
        st.write("**Step 1: Select paper type**")

        labels = [PAPER_TYPE_LABELS[pt] for pt in PAPER_TYPES]
        type_key = f"{self.CACHE_PREFIX}_paper_type"
        prev_type = st.session_state.get(type_key)

        col1, col2 = st.columns([2, 3])
        with col1:
            selected_label = st.selectbox(
                "Paper type",
                options=["— select —"] + labels,
                key=f"{self.CACHE_PREFIX}_paper_type_select",
            )
        with col2:
            if selected_label and selected_label != "— select —":
                paper_type = PAPER_TYPES[labels.index(selected_label)]
                st.session_state[type_key] = paper_type

                # Show standard sections for this type
                default_secs = SECTION_DEFAULTS.get(paper_type, DEFAULT_SECTIONS)
                st.caption(f"Standard sections for this type: {' · '.join(default_secs)}")

    # ------------------------------------------------------------------
    # Auto-detect flow
    # ------------------------------------------------------------------

    def _render_auto_detect_flow(self, files: Optional[Dict[str, bytes]], paper_type: str, figures_external: bool = False):
        if not files:
            st.info("Upload a PDF, .tex, or .txt file using the Document Uploader above.")
            return

        file_keys = list(files.keys())
        manuscript = st.selectbox("Select manuscript", options=file_keys, key=f"{self.CACHE_PREFIX}_select")
        if manuscript is None:
            return

        scan_state_key = f"{self.CACHE_PREFIX}_detected_{manuscript}"
        text_state_key = f"{self.CACHE_PREFIX}_text_{manuscript}"

        # --- Phase 1: Scan ---
        if st.button("Scan for Sections", key=f"{self.CACHE_PREFIX}_scan_{manuscript}"):
            with st.spinner("Extracting text..."):
                paper_text = decode_file(manuscript, files[manuscript], warn_fn=st.warning)
            with st.spinner("Detecting section headers..."):
                detected = detect_sections(paper_text, self.llm)
            st.session_state[scan_state_key] = detected
            st.session_state[text_state_key] = paper_text
            # Clear stale preview and hierarchy caches from any previous scan
            st.session_state.pop(f"{self.CACHE_PREFIX}_preview_{manuscript}", None)
            st.session_state.pop(f"{self.CACHE_PREFIX}_hierarchy_{manuscript}", None)
            st.success(f"Found {len(detected)} section(s).")

        # --- Phase 1.5: Missing section search ---
        detected = st.session_state.get(scan_state_key)
        paper_text = st.session_state.get(text_state_key, "")

        if detected:
            detected = self._normalize(detected)
            st.session_state[scan_state_key] = detected

            st.write("**Did we miss any sections?**")
            col1, col2 = st.columns([3, 1])
            with col1:
                hint = st.text_input(
                    "Section hint",
                    key=f"{self.CACHE_PREFIX}_missing_{manuscript}",
                    placeholder="e.g., Discussion"
                )
            with col2:
                st.write("")
                st.write("")
                if st.button("Search", key=f"{self.CACHE_PREFIX}_search_missing_{manuscript}"):
                    if hint and paper_text:
                        found = search_missing_section(paper_text, hint)
                        if found:
                            inserted = False
                            for i, sec in enumerate(detected):
                                if sec.get("line_idx", 999999) > found.get("line_idx", 0):
                                    detected.insert(i, found)
                                    inserted = True
                                    break
                            if not inserted:
                                detected.append(found)
                            st.session_state[scan_state_key] = detected
                            st.success(f"Added: {found['text']}")
                            st.rerun()
                        else:
                            st.warning(f"Could not find '{hint}'. Try a different hint.")

        # --- Phase 2: Section selection ---
        detected = st.session_state.get(scan_state_key)
        if not detected:
            return

        # Cache hierarchy so group_subsections (which may call LLM) only runs once per scan
        hierarchy_key = f"{self.CACHE_PREFIX}_hierarchy_{manuscript}"
        if hierarchy_key not in st.session_state:
            st.session_state[hierarchy_key] = group_subsections(detected, self.llm)
        hierarchy_result = st.session_state[hierarchy_key]
        hierarchy = hierarchy_result.get("hierarchy", {})
        style_info = hierarchy_result.get("style_info", {})
        top_level_from_analysis = hierarchy_result.get("top_level", [])

        subsection_texts = set()
        for parent, children in hierarchy.items():
            subsection_texts.update(children)

        if top_level_from_analysis:
            top_level_sections = [sec for sec in detected if sec["text"] in top_level_from_analysis]
        else:
            top_level_sections = [sec for sec in detected if sec["text"] not in subsection_texts]

        is_suggestion = all(d.get("type") == "suggestion" for d in detected)
        if is_suggestion:
            st.info("Could not auto-detect sections. Showing default suggestions.")
        else:
            self._render_style_info(style_info, hierarchy, top_level_sections, subsection_texts)

        # Section checkboxes
        display_sections = top_level_sections if not is_suggestion else detected
        actions = {sec_text: "keep" for sec_text in subsection_texts}
        eval_selected = {}

        # Preview extraction
        preview_key = f"{self.CACHE_PREFIX}_preview_{manuscript}"
        if paper_text and preview_key not in st.session_state:
            all_names = [sec["text"] for sec in detected]
            with st.spinner("Extracting section previews..."):
                preview_sections = extract_sections_from_text(paper_text, detected, all_names)
            st.session_state[preview_key] = preview_sections
        preview_sections = st.session_state.get(preview_key, {})

        st.write("---")
        col_h1, col_h2, col_h3 = st.columns([0.5, 3, 2])
        with col_h1:
            st.caption("Evaluate")
        with col_h2:
            st.caption("Section")
        with col_h3:
            st.caption("Action")

        section_names_list = [sec["text"] for sec in display_sections]

        for i, sec in enumerate(display_sections):
            sec_text = sec["text"]
            sec_type = sec.get("type", "")
            display_label = f"{sec_text}  ({sec_type})" if sec_type and sec_type not in ("other", "suggestion") else sec_text

            col1, col2, col3 = st.columns([0.5, 3, 2])
            with col1:
                eval_selected[sec_text] = st.checkbox(
                    "Evaluate", value=True,
                    key=f"{self.CACHE_PREFIX}_eval_{sec_text}_{manuscript}",
                    label_visibility="hidden",
                )
            with col2:
                st.markdown(f"**{display_label}**")
            with col3:
                other_options = [s for s in section_names_list if s != sec_text]
                merge_options = ["keep", "remove"] + [f"merge → {o}" for o in other_options]
                action = st.selectbox(
                    "Action",
                    options=merge_options,
                    key=f"{self.CACHE_PREFIX}_action_{sec_text}_{manuscript}",
                    label_visibility="collapsed",
                )
                actions[sec_text] = action

            # Full section text preview (including any subsections), matching old behaviour
            section_preview_text = preview_sections.get(sec_text, "")
            # Append subsection text into the parent preview
            children_texts = hierarchy.get(sec_text, [])
            for child_text in children_texts:
                child_preview = preview_sections.get(child_text, "")
                if child_preview:
                    section_preview_text += "\n\n" + child_preview

            if section_preview_text:
                word_count = len(section_preview_text.split())
                char_count = len(section_preview_text)
                with st.expander(f"👁️ Preview extracted text ({word_count:,} words, {char_count:,} chars)"):
                    if children_texts:
                        child_names = [c for c in children_texts if preview_sections.get(c)]
                        if child_names:
                            st.info(f"Preview includes subsections: {', '.join(child_names)}")
                    if char_count <= 3000:
                        st.text_area(
                            "Raw extracted text:",
                            section_preview_text,
                            height=300,
                            key=f"{self.CACHE_PREFIX}_preview_text_{manuscript}_{i}",
                            disabled=True,
                        )
                    else:
                        show_full_key = f"{self.CACHE_PREFIX}_show_full_{manuscript}_{i}"
                        st.session_state.setdefault(show_full_key, False)
                        if st.session_state[show_full_key]:
                            st.text_area(
                                "Raw extracted text (full):",
                                section_preview_text,
                                height=500,
                                key=f"{self.CACHE_PREFIX}_preview_text_full_{manuscript}_{i}",
                                disabled=True,
                            )
                            if st.button("Show less", key=f"{self.CACHE_PREFIX}_show_less_{manuscript}_{i}"):
                                st.session_state[show_full_key] = False
                                st.rerun()
                        else:
                            st.text_area(
                                "Raw extracted text (preview — first 3000 chars):",
                                section_preview_text[:3000] + "\n\n... [truncated]",
                                height=300,
                                key=f"{self.CACHE_PREFIX}_preview_text_short_{manuscript}_{i}",
                                disabled=True,
                            )
                            if st.button("Show full text", key=f"{self.CACHE_PREFIX}_show_full_btn_{manuscript}_{i}"):
                                st.session_state[show_full_key] = True
                                st.rerun()
            else:
                with st.expander("👁️ Preview extracted text"):
                    st.warning("⚠️ No text could be extracted for this section. The section header was detected but content extraction failed. Consider removing this section or merging it.")

        # --- Phase 3: Context ---
        self._render_context_options(paper_text, manuscript)

        # --- Phase 4: Evaluate ---
        if st.button("Evaluate Selected Sections", key=f"{self.CACHE_PREFIX}_evaluate_{manuscript}", type="primary"):
            to_evaluate = [s for s in display_sections if eval_selected.get(s["text"]) and actions.get(s["text"]) != "remove"]
            if not to_evaluate:
                st.warning("No sections selected. Check at least one section to evaluate.")
                return

            # Merge sections
            merged = self._apply_merges(to_evaluate, actions, paper_text, detected)

            # Run evaluation
            paper_context = st.session_state.get(f"{self.CACHE_PREFIX}_paper_context_{manuscript}", "")
            results = self._run_evaluation(merged, paper_type, paper_context, figures_external)
            st.session_state[f"{self.CACHE_PREFIX}_results_{manuscript}"] = results

        # Display results if available
        results = st.session_state.get(f"{self.CACHE_PREFIX}_results_{manuscript}")
        if results:
            self._render_results(results, paper_type)

    # ------------------------------------------------------------------
    # Freeform (manual paste) flow
    # ------------------------------------------------------------------

    def _render_freeform_flow(self, paper_type: str, figures_external: bool = False):
        default_secs = SECTION_DEFAULTS.get(paper_type, DEFAULT_SECTIONS)
        freeform_key = f"{self.CACHE_PREFIX}_freeform"

        st.write("**Step 3: Paste section text**")
        st.write("Paste the text for each section you want evaluated. Leave blank to skip.")

        # Context
        st.write("---")
        with st.expander("Optional: provide full paper or abstract for context", expanded=False):
            context_mode = st.radio(
                "Context type",
                ["Abstract only", "Full paper"],
                key=f"{freeform_key}_context_mode",
                horizontal=True,
            )
            context_text = st.text_area(
                "Paste abstract or full paper here:",
                height=150,
                key=f"{freeform_key}_context_text",
            )
            paper_context = ""
            if context_text.strip():
                if context_mode == "Full paper":
                    if st.button("Summarize paper for context", key=f"{freeform_key}_summarize"):
                        with st.spinner("Summarizing paper..."):
                            paper_context = self._evaluator.generate_context_summary(context_text)
                        st.session_state[f"{freeform_key}_paper_context"] = paper_context
                        st.success("Paper summarized.")
                else:
                    paper_context = context_text.strip()
                    st.session_state[f"{freeform_key}_paper_context"] = paper_context
        paper_context = st.session_state.get(f"{freeform_key}_paper_context", "")

        # Section text inputs
        st.write("---")
        st.write("**Sections**")

        # Allow adding custom sections beyond defaults
        custom_sections = st.session_state.get(f"{freeform_key}_custom_sections", [])
        all_sections = list(default_secs) + custom_sections

        new_sec = st.text_input(
            "Add a custom section",
            key=f"{freeform_key}_new_section",
            placeholder="e.g., Stylized Facts"
        )
        if st.button("Add section", key=f"{freeform_key}_add_section"):
            if new_sec.strip() and new_sec.strip() not in all_sections:
                custom_sections.append(new_sec.strip())
                st.session_state[f"{freeform_key}_custom_sections"] = custom_sections
                st.rerun()

        section_texts = {}
        eval_flags = {}
        for sec_name in all_sections:
            col1, col2 = st.columns([0.3, 3])
            with col1:
                eval_flags[sec_name] = st.checkbox(
                    "Evaluate",
                    value=False,
                    key=f"{freeform_key}_flag_{sec_name}",
                )
            with col2:
                with st.expander(sec_name, expanded=False):
                    text = st.text_area(
                        "Section text",
                        height=150,
                        key=f"{freeform_key}_text_{sec_name}",
                        label_visibility="collapsed",
                    )
                    section_texts[sec_name] = text.strip()

        to_evaluate = {name: text for name, text in section_texts.items()
                       if eval_flags.get(name) and text}

        if not to_evaluate:
            st.info("Check 'Evaluate' and paste text for at least one section, then click Evaluate.")

        if st.button("Evaluate Selected Sections", key=f"{freeform_key}_evaluate", type="primary"):
            if not to_evaluate:
                st.warning("No sections with text selected.")
                return
            results = self._run_evaluation(to_evaluate, paper_type, paper_context, figures_external)
            st.session_state[f"{freeform_key}_results"] = results

        results = st.session_state.get(f"{freeform_key}_results")
        if results:
            self._render_results(results, paper_type)

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _render_context_options(self, paper_text: str, manuscript: str):
        context_key = f"{self.CACHE_PREFIX}_paper_context_{manuscript}"
        st.write("---")
        with st.expander("Optional: paper context for the LLM", expanded=False):
            if paper_text:
                if st.button("Auto-summarize paper for context", key=f"{self.CACHE_PREFIX}_ctx_summarize_{manuscript}"):
                    with st.spinner("Summarizing paper..."):
                        summary = self._evaluator.generate_context_summary(paper_text)
                    st.session_state[context_key] = summary
                    st.success("Summarized.")
            manual_ctx = st.text_area(
                "Or type/paste a brief paper description:",
                value=st.session_state.get(context_key, ""),
                height=100,
                key=f"{self.CACHE_PREFIX}_ctx_manual_{manuscript}",
            )
            if manual_ctx.strip():
                st.session_state[context_key] = manual_ctx.strip()

    # ------------------------------------------------------------------
    # Evaluation runner
    # ------------------------------------------------------------------

    def _apply_merges(
        self,
        to_evaluate: List[Dict],
        actions: Dict[str, str],
        paper_text: str,
        detected: List[Dict],
    ) -> Dict[str, str]:
        """Resolve merge actions and return {section_name: section_text} dict."""
        all_names = [sec["text"] for sec in detected]
        extracted = extract_sections_from_text(paper_text, detected, all_names)

        # Map merge targets
        merged: Dict[str, str] = {}
        merge_map: Dict[str, str] = {}

        for sec in to_evaluate:
            sec_text = sec["text"]
            action = actions.get(sec_text, "keep")
            if action.startswith("merge → "):
                target = action[len("merge → "):]
                merge_map[sec_text] = target

        for sec in to_evaluate:
            sec_text = sec["text"]
            action = actions.get(sec_text, "keep")
            # Only include sections that are kept (not removed, not merging into another)
            if action != "remove" and not action.startswith("merge → "):
                body = extracted.get(sec_text, "")
                # Append any sections that are merging into this one
                for source, target in merge_map.items():
                    if target == sec_text:
                        body = body + "\n\n" + extracted.get(source, "")
                if body.strip():
                    merged[sec_text] = body.strip()

        return merged

    def _run_evaluation(
        self,
        section_texts: Dict[str, str],
        paper_type: str,
        paper_context: str = "",
        figures_external: bool = False,
    ) -> Dict[str, Any]:
        results = {}
        progress = st.progress(0)
        total = len(section_texts)
        for i, (name, text) in enumerate(section_texts.items()):
            with st.spinner(f"Evaluating: {name}..."):
                result = self._evaluator.evaluate_section(
                    section_name=name,
                    section_text=text,
                    paper_type=paper_type,
                    paper_context=paper_context,
                    figures_external=figures_external,
                )
            results[name] = result
            progress.progress((i + 1) / total)
        progress.empty()
        return results

    # ------------------------------------------------------------------
    # Results rendering
    # ------------------------------------------------------------------

    def _render_results(self, results: Dict[str, Any], paper_type: str):
        st.write("---")
        st.subheader("Evaluation Results")

        # Overall summary
        section_scores = {name: res["section_score"] for name, res in results.items()}
        overall = compute_overall_score(section_scores, paper_type)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Score", f"{overall['overall_score']:.1f} / 5.0")
        with col2:
            st.metric("Publication Readiness", overall["publication_readiness"])
        with col3:
            st.metric("Sections Evaluated", len(results))

        st.progress(score_bar_html(overall["overall_score"]))

        # Overall assessment from LLM
        if st.button("Generate Overall Assessment", key="se_v3_overall_assessment"):
            with st.spinner("Generating overall assessment..."):
                overall_text = self._evaluator.generate_overall_assessment(results, paper_type)
            st.session_state["se_v3_overall_text"] = overall_text

        overall_text = st.session_state.get("se_v3_overall_text")
        if overall_text:
            st.markdown(overall_text)

        # Per-section results
        st.write("---")
        st.write("**Section-by-section results**")

        for section_name, result in results.items():
            sc = result.get("section_score", {})
            raw = sc.get("raw_score", 3.0)

            fatal_triggered = sc.get("fatal_flaw_triggered", False)
            fatal_label = " ⚠️ FATAL FLAW" if fatal_triggered else ""
            with st.expander(
                f"**{section_name}** — Score: {raw:.1f}/5.0 (adjusted: {sc.get('adjusted_score', raw):.1f}){fatal_label}",
                expanded=False,
            ):
                if fatal_triggered:
                    flaw_crits = sc.get("fatal_flaw_criteria", [])
                    st.error(
                        f"**Fatal flaw detected.** The following critical criterion scored ≤ 1.5, "
                        f"capping the section score at 2.5: **{', '.join(flaw_crits)}**. "
                        "This indicates a fundamental issue that must be resolved before the section can score higher."
                    )
                st.progress(score_bar_html(raw))

                # Qualitative assessment
                st.markdown("### Qualitative Assessment")
                st.write(result.get("qualitative_assessment", ""))

                # Criteria breakdown
                st.markdown("### Criteria Breakdown")
                criteria_evals = result.get("criteria_evaluations", [])

                for ev in criteria_evals:
                    criterion = ev.get("criterion", "")
                    score = ev.get("score", 3)
                    weight = ev.get("weight", 0.25)
                    description = ev.get("description", "")
                    justification = ev.get("justification", "")
                    q1 = ev.get("quote_1", {})
                    q2 = ev.get("quote_2", {})

                    score_color = "🟢" if score >= 4 else "🟡" if score == 3 else "🔴"
                    is_fatal = ev.get("is_fatal_criterion", False)
                    fatal_badge = " <span style='color:#c0392b;font-size:0.8em;font-weight:bold'>CRITICAL</span>" if is_fatal else ""
                    st.markdown(
                        f"**{criterion}**{fatal_badge} {score_color} {score}/5 "
                        f"<span style='color:gray;font-size:0.85em'>(weight {weight:.0%})</span>",
                        unsafe_allow_html=True,
                    )
                    if description:
                        st.caption(description)
                    if justification:
                        st.write(justification)

                    # Quotes
                    if q1.get("text"):
                        valid_icon = "✓" if q1.get("valid") else "~"
                        supports = "supports" if q1.get("supports_assessment") else "complicates"
                        st.markdown(
                            f"<blockquote style='border-left:3px solid #ccc;padding-left:8px;color:#555'>"
                            f"{valid_icon} [{supports}] \"{q1['text']}\""
                            f"</blockquote>",
                            unsafe_allow_html=True,
                        )
                    if q2.get("text"):
                        valid_icon = "✓" if q2.get("valid") else "~"
                        supports = "supports" if q2.get("supports_assessment") else "complicates"
                        st.markdown(
                            f"<blockquote style='border-left:3px solid #ccc;padding-left:8px;color:#555'>"
                            f"{valid_icon} [{supports}] \"{q2['text']}\""
                            f"</blockquote>",
                            unsafe_allow_html=True,
                        )

                    st.write("")

                # Improvements
                improvements = result.get("improvements", [])
                if improvements:
                    st.markdown("### Priority Improvements")
                    for imp in sorted(improvements, key=lambda x: x.get("priority", 99)):
                        priority = imp.get("priority", "")
                        suggestion = imp.get("suggestion", "")
                        rationale = imp.get("rationale", "")
                        st.write(f"**{priority}.** {suggestion}")
                        if rationale:
                            st.caption(rationale)

                # Weight breakdown visualization
                st.markdown("### Score Breakdown")
                breakdown = sc.get("criteria_breakdown", {})
                weights = sc.get("weight_breakdown", {})
                for crit, crit_score in breakdown.items():
                    w = weights.get(crit, 0.25)
                    st.write(f"- {crit}: {crit_score}/5 (weight {w:.0%})")
                importance = sc.get("importance_multiplier", 1.0)
                if importance != 1.0:
                    st.caption(f"Section importance multiplier: ×{importance:.1f} (reflects this section's weight in {PAPER_TYPE_LABELS.get(paper_type, paper_type)} papers)")

        # Download section
        st.write("---")
        st.subheader("Download Report")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Generate markdown report
            md_report = self._build_markdown_report(results, paper_type, overall_text)
            st.download_button(
                label="📄 Download as Markdown",
                data=md_report,
                file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

        with col2:
            # Generate PDF report
            try:
                pdf_data = self.generate_pdf_report(results, paper_type, overall_text or "")
                st.download_button(
                    label="📕 Download as PDF",
                    data=pdf_data,
                    file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating PDF: {e}")

        with col3:
            # Generate CSV report (one row per criterion — useful for benchmarking)
            csv_data = self._build_csv_report(results, paper_type)
            st.download_button(
                label="📊 Download as CSV (scores)",
                data=csv_data,
                file_name=f"evaluation_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="One row per criterion. Useful for tracking scores over time and benchmarking across papers.",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(sections: List[Dict]) -> List[Dict]:
        return [
            {
                "text": s.get("text", "Unknown"),
                "type": s.get("type", "other"),
                "line_idx": s.get("line_idx", i),
            }
            for i, s in enumerate(sections)
        ]

    def _render_style_info(self, style_info, hierarchy, top_level_sections, subsection_texts):
        primary_style = style_info.get("primary_style", "unknown")
        subsection_style = style_info.get("subsection_style", "none")
        confidence = style_info.get("confidence", 0)

        style_display = {
            "arabic": "Arabic numerals (1, 2, 3…)",
            "roman_upper": "Roman numerals (I, II, III…)",
            "roman_lower": "Roman numerals (i, ii, iii…)",
            "letter_upper": "Letters (A, B, C…)",
            "letter_lower": "Letters (a, b, c…)",
            "numeric_dot": "Decimal numbering (1.1, 1.2…)",
            "mixed": "Mixed styles",
            "none": "No clear pattern",
        }

        st.write(f"**Detected numbering style:** {style_display.get(primary_style, primary_style)}")
        if subsection_style != "none":
            st.write(f"**Subsection style:** {style_display.get(subsection_style, subsection_style)}")
        if confidence < 0.5:
            st.warning(f"Low confidence ({confidence:.0%}) in style detection. Using LLM-assisted hierarchy.")

        if hierarchy:
            with st.expander("View complete section hierarchy"):
                for parent, children in hierarchy.items():
                    st.write(f"**{parent}**")
                    for child in children:
                        st.write(f"  └─ {child}")
                st.info(
                    f"Showing {len(top_level_sections)} top-level sections. "
                    f"{len(subsection_texts)} subsections hidden but included during evaluation."
                )

    def generate_pdf_report(self, results: Dict[str, Any], paper_type: str, overall_text: str = "") -> bytes:
        """Generate a PDF report from the evaluation results."""
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 16)
                self.cell(0, 10, 'Manuscript Evaluation Report', 0, 1, 'C')
                self.set_font('Arial', 'I', 10)
                self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

            def chapter_title(self, title):
                self.set_font('Arial', 'B', 14)
                self.cell(0, 10, title, 0, 1, 'L')
                self.ln(2)

            def section_title(self, title):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 8, title, 0, 1, 'L')
                self.ln(1)

            def body_text(self, text):
                self.set_font('Arial', '', 11)
                # Handle encoding issues
                text = text.encode('latin-1', 'replace').decode('latin-1')
                self.multi_cell(0, 6, text)
                self.ln(2)

            def bullet_point(self, text):
                self.set_font('Arial', '', 11)
                text = text.encode('latin-1', 'replace').decode('latin-1')
                self.cell(10, 6, '-', 0, 0)
                self.multi_cell(0, 6, text)

        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()

        # Overall Summary
        pdf.chapter_title('Overall Summary')
        section_scores = {name: res["section_score"] for name, res in results.items()}
        overall_stats = compute_overall_score(section_scores, paper_type)

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, f'Overall Score: {overall_stats["overall_score"]:.1f} / 5.0', 0, 1)
        pdf.cell(0, 8, f'Publication Readiness: {overall_stats["publication_readiness"]}', 0, 1)
        pdf.cell(0, 8, f'Sections Evaluated: {len(results)}', 0, 1)
        pdf.cell(0, 8, f'Paper Type: {PAPER_TYPE_LABELS.get(paper_type, paper_type)}', 0, 1)
        pdf.ln(5)

        # Overall Assessment
        if overall_text:
            pdf.chapter_title('Overall Assessment')
            pdf.body_text(overall_text)
            pdf.ln(5)

        # Section Evaluations
        pdf.chapter_title('Section Evaluations')

        for section_name, result in results.items():
            sc = result.get("section_score", {})
            raw_score = sc.get("raw_score", 3.0)
            adjusted_score = sc.get("adjusted_score", raw_score)

            pdf.section_title(f'{section_name} - Score: {raw_score:.1f}/5.0 (adjusted: {adjusted_score:.1f})')

            # Qualitative Assessment
            qual = result.get("qualitative_assessment", "")
            if qual:
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 6, 'Qualitative Assessment:', 0, 1)
                pdf.body_text(qual)

            # Criteria Breakdown
            criteria_evals = result.get("criteria_evaluations", [])
            if criteria_evals:
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 6, 'Criteria Breakdown:', 0, 1)
                pdf.set_font('Arial', '', 11)

                for ev in criteria_evals:
                    criterion = ev.get("criterion", "")
                    score = ev.get("score", 3)
                    weight = ev.get("weight", 0.25)
                    justification = ev.get("justification", "")

                    pdf.bullet_point(f'{criterion}: {score}/5 (weight {weight:.0%})')
                    if justification:
                        pdf.set_x(20)
                        pdf.multi_cell(0, 5, justification.encode('latin-1', 'replace').decode('latin-1'))
                        pdf.ln(1)

            # Improvements
            improvements = result.get("improvements", [])
            if improvements:
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 6, 'Priority Improvements:', 0, 1)
                pdf.set_font('Arial', '', 11)

                for imp in sorted(improvements, key=lambda x: x.get("priority", 99)):
                    priority = imp.get("priority", "")
                    suggestion = imp.get("suggestion", "")
                    rationale = imp.get("rationale", "")

                    text = f'{priority}. {suggestion}'
                    if rationale:
                        text += f' ({rationale})'
                    pdf.bullet_point(text)

            pdf.ln(5)

        # Return PDF as bytes
        return pdf.output(dest='S').encode('latin-1')

    def _build_csv_report(self, results: Dict[str, Any], paper_type: str) -> str:
        """Generate a CSV with one row per criterion across all sections."""
        section_scores = {name: res["section_score"] for name, res in results.items()}
        overall = compute_overall_score(section_scores, paper_type)

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "section", "section_raw_score", "section_adjusted_score", "section_importance_multiplier",
            "criterion", "criterion_score", "criterion_weight", "criterion_weighted_contribution",
            "justification", "quote_1", "quote_1_valid", "quote_2", "quote_2_valid",
            "has_fatal_flaw", "fatal_flaw_triggered",
            "paper_type", "overall_score", "publication_readiness",
            "eval_timestamp", "model_version",
        ])

        overall_score = overall["overall_score"]
        readiness = overall["publication_readiness"]

        for section_name, result in results.items():
            sc = result.get("section_score", {})
            raw = sc.get("raw_score", 3.0)
            adj = sc.get("adjusted_score", raw)
            imp = sc.get("importance_multiplier", 1.0)
            timestamp = result.get("eval_timestamp", "")
            model_ver = result.get("model_version", "")
            fatal_triggered = sc.get("fatal_flaw_triggered", False)

            for ev in result.get("criteria_evaluations", []):
                score = ev.get("score", 3)
                weight = ev.get("weight", 0.25)
                q1 = ev.get("quote_1", {})
                q2 = ev.get("quote_2", {})
                is_fatal = ev.get("is_fatal_criterion", False)
                writer.writerow([
                    section_name, raw, adj, imp,
                    ev.get("criterion", ""), score, weight, round(score * weight, 4),
                    ev.get("justification", ""),
                    q1.get("text", ""), q1.get("valid", ""),
                    q2.get("text", ""), q2.get("valid", ""),
                    is_fatal, fatal_triggered,
                    paper_type, overall_score, readiness,
                    timestamp, model_ver,
                ])

        return output.getvalue()

    def _build_markdown_report(self, results: Dict[str, Any], paper_type: str, overall_text: str = "") -> str:
        """Generate a markdown report from the evaluation results."""
        section_scores = {name: res["section_score"] for name, res in results.items()}
        overall_stats = compute_overall_score(section_scores, paper_type)

        md = f"""# Manuscript Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Overall Summary

- **Overall Score:** {overall_stats["overall_score"]:.1f} / 5.0
- **Publication Readiness:** {overall_stats["publication_readiness"]}
- **Sections Evaluated:** {len(results)}
- **Paper Type:** {PAPER_TYPE_LABELS.get(paper_type, paper_type)}

"""

        if overall_text:
            md += f"""## Overall Assessment

{overall_text}

"""

        md += "## Section Evaluations\n\n"

        for section_name, result in results.items():
            sc = result.get("section_score", {})
            raw_score = sc.get("raw_score", 3.0)
            adjusted_score = sc.get("adjusted_score", raw_score)

            md += f"### {section_name} — Score: {raw_score:.1f}/5.0 (adjusted: {adjusted_score:.1f})\n\n"

            # Qualitative Assessment
            qual = result.get("qualitative_assessment", "")
            if qual:
                md += f"**Qualitative Assessment:**\n\n{qual}\n\n"

            # Criteria Breakdown
            criteria_evals = result.get("criteria_evaluations", [])
            if criteria_evals:
                md += "**Criteria Breakdown:**\n\n"
                for ev in criteria_evals:
                    criterion = ev.get("criterion", "")
                    score = ev.get("score", 3)
                    weight = ev.get("weight", 0.25)
                    justification = ev.get("justification", "")

                    md += f"- **{criterion}:** {score}/5 (weight {weight:.0%})\n"
                    if justification:
                        md += f"  - {justification}\n"
                md += "\n"

            # Improvements
            improvements = result.get("improvements", [])
            if improvements:
                md += "**Priority Improvements:**\n\n"
                for imp in sorted(improvements, key=lambda x: x.get("priority", 99)):
                    priority = imp.get("priority", "")
                    suggestion = imp.get("suggestion", "")
                    rationale = imp.get("rationale", "")

                    md += f"{priority}. {suggestion}\n"
                    if rationale:
                        md += f"   - {rationale}\n"
                md += "\n"

            md += "---\n\n"

        return md

    # ------------------------------------------------------------------
    # Wrapper methods for compatibility with app_demo.py
    # ------------------------------------------------------------------

    def evaluate_section(self, section_name: str, section_text: str, paper_type: str = "empirical"):
        """Wrapper to expose evaluator.evaluate_section for direct calls."""
        return self._evaluator.evaluate_section(section_name, section_text, paper_type)

    def generate_overall_assessment(self, sections: Dict[str, str], evaluations: Dict[str, Any], paper_type: str = "empirical"):
        """Wrapper to expose evaluator.generate_overall_assessment for direct calls."""
        # Convert evaluations to results format expected by evaluator
        results = {name: eval_data for name, eval_data in evaluations.items()}
        return self._evaluator.generate_overall_assessment(results, paper_type)

    def display_results(self, sections: Dict[str, str], evaluations: Dict[str, Any], overall_assessment: str, paper_type: str = "empirical"):
        """Display evaluation results in Streamlit UI."""
        # Use the existing _render_results method
        self._render_results(evaluations, paper_type)

        # Display overall assessment
        if overall_assessment:
            st.markdown("---")
            st.markdown("### 📊 Overall Assessment")
            st.markdown(overall_assessment)
