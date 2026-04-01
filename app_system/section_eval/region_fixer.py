"""
Interactive region fixer UI for detected problematic math/table regions.

This module provides a Streamlit UI component that allows users to review
and fix problematic regions detected during PDF extraction.
"""

import streamlit as st
from typing import Dict, List, Optional, Tuple
from .math_cleanup import MathRegion, detect_math_regions, build_cleanup_prompt


def render_region_fixer(
    text: str,
    manuscript_name: str,
    cache_prefix: str,
    llm_query_fn,
    min_confidence: float = 0.5
) -> str:
    """
    Render an interactive UI for fixing problematic regions in extracted text.

    Detects equations and tables with extraction issues, then allows users to:
    - Auto-clean with LLM
    - Manually paste corrected text
    - Upload an image (for OCR)
    - Skip (keep original)

    Args:
        text: Raw extracted text
        manuscript_name: Name of the manuscript (for cache keys)
        cache_prefix: Prefix for session state keys
        llm_query_fn: Function to call LLM (e.g., single_query)
        min_confidence: Minimum confidence score to show a region (0-1)

    Returns:
        Modified text with user fixes applied
    """
    # Detect problematic regions
    regions = detect_math_regions(text)
    regions = [r for r in regions if r.confidence >= min_confidence]

    if not regions:
        st.success("✅ No extraction issues detected - text quality looks good!")
        st.info("💡 Proceeding directly to section detection...")
        # Auto-advance to next phase
        st.session_state[f"{cache_prefix}_fixes_applied_{manuscript_name}"] = True
        st.session_state[f"{cache_prefix}_fixed_text_{manuscript_name}"] = text
        st.rerun()
        return text

    # Sort by confidence (most broken first)
    regions = sorted(regions, key=lambda r: r.confidence, reverse=True)

    # State management
    fixes_key = f"{cache_prefix}_region_fixes_{manuscript_name}"
    if fixes_key not in st.session_state:
        st.session_state[fixes_key] = {}
    fixes = st.session_state[fixes_key]

    # UI Header
    st.warning(f"⚠️ Detected **{len(regions)} problematic region(s)** in extraction")
    st.caption(
        "Review and fix critical content (especially equations). "
        "You can auto-clean with LLM, paste corrected text, or skip."
    )

    # Show/hide toggle
    show_fixer = st.checkbox(
        f"📋 Review {len(regions)} region(s)",
        value=False,
        key=f"{cache_prefix}_show_fixer_{manuscript_name}",
        help="Expand to review and fix problematic regions"
    )

    if not show_fixer:
        st.info("💡 Skipping review. Click above to review and fix regions.")
        return text

    st.markdown("---")

    # Render each region
    for i, region in enumerate(regions, 1):
        _render_region_editor(
            region=region,
            region_idx=i,
            fixes=fixes,
            cache_prefix=cache_prefix,
            manuscript_name=manuscript_name,
            llm_query_fn=llm_query_fn
        )

    # Apply fixes button
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        num_fixed = len([f for f in fixes.values() if f.get("action") != "skip"])
        st.caption(f"📊 {num_fixed} region(s) will be modified")

    with col2:
        if st.button("✅ Apply Fixes and Continue", type="primary",
                    key=f"{cache_prefix}_apply_fixes_{manuscript_name}"):
            modified_text = _apply_fixes(text, regions, fixes)
            # Signal that review is complete
            st.session_state[f"{cache_prefix}_fixes_applied_{manuscript_name}"] = True
            st.session_state[f"{cache_prefix}_fixed_text_{manuscript_name}"] = modified_text
            st.success(f"✅ Applied {num_fixed} fix(es)")
            st.rerun()

    with col3:
        if st.button("⏭️ Skip All", key=f"{cache_prefix}_skip_all_{manuscript_name}"):
            # Signal that review is complete (no changes)
            st.session_state[f"{cache_prefix}_fixes_applied_{manuscript_name}"] = True
            st.session_state[f"{cache_prefix}_fixed_text_{manuscript_name}"] = text
            st.info("Continuing with original extraction")
            st.rerun()

    # Not done yet - return original text
    return text


def _render_region_editor(
    region: MathRegion,
    region_idx: int,
    fixes: Dict,
    cache_prefix: str,
    manuscript_name: str,
    llm_query_fn
):
    """Render editor for a single region."""
    region_id = f"region_{region_idx}"

    # Initialize fix state for this region
    if region_id not in fixes:
        fixes[region_id] = {
            "action": "skip",  # skip, llm, manual, image
            "content": None,
            "original_start": region.start_idx,
            "original_end": region.end_idx
        }

    fix = fixes[region_id]

    # Region header
    with st.expander(
        f"**Region #{region_idx}** — {region.region_type.upper()} "
        f"(confidence: {region.confidence:.2f})",
        expanded=(region_idx <= 3)  # Expand first 3
    ):
        # Show original content
        st.markdown("**📄 Current Extraction:**")
        st.code(region.content, language=None)

        # Action selector
        st.markdown("**🔧 How to fix this region:**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button(
                "🤖 Auto-clean",
                key=f"{cache_prefix}_llm_{region_idx}_{manuscript_name}",
                help="Use LLM to clean up formatting",
                use_container_width=True
            ):
                with st.spinner("Cleaning with LLM..."):
                    prompt = build_cleanup_prompt(region)
                    try:
                        cleaned = llm_query_fn(prompt).strip()
                        fix["action"] = "llm"
                        fix["content"] = cleaned
                        st.rerun()
                    except Exception as e:
                        st.error(f"LLM error: {e}")

        with col2:
            if st.button(
                "✏️ Paste text",
                key=f"{cache_prefix}_manual_{region_idx}_{manuscript_name}",
                help="Manually paste corrected text",
                use_container_width=True
            ):
                fix["action"] = "manual"
                st.rerun()

        with col3:
            if st.button(
                "🖼️ Upload image",
                key=f"{cache_prefix}_image_{region_idx}_{manuscript_name}",
                help="Upload screenshot to extract text",
                use_container_width=True
            ):
                fix["action"] = "image"
                st.rerun()

        with col4:
            if st.button(
                "⏭️ Skip",
                key=f"{cache_prefix}_skip_{region_idx}_{manuscript_name}",
                help="Keep original extraction",
                use_container_width=True
            ):
                fix["action"] = "skip"
                fix["content"] = None
                st.rerun()

        # Show input UI based on action
        if fix["action"] == "llm" and fix["content"]:
            st.markdown("**✨ LLM Cleaned Version:**")
            st.success(fix["content"])
            st.caption("✅ This will replace the original text")

        elif fix["action"] == "manual":
            st.markdown("**✏️ Paste Corrected Text:**")
            manual_text = st.text_area(
                "Corrected text",
                value=fix.get("content", region.content),
                height=150,
                key=f"{cache_prefix}_manual_text_{region_idx}_{manuscript_name}",
                help="Paste the corrected equation or table here"
            )
            if st.button("💾 Save", key=f"{cache_prefix}_save_manual_{region_idx}_{manuscript_name}"):
                fix["content"] = manual_text
                st.success("✅ Saved!")
                st.rerun()

        elif fix["action"] == "image":
            st.markdown("**🖼️ Upload Screenshot:**")
            uploaded_file = st.file_uploader(
                "Upload image",
                type=["png", "jpg", "jpeg"],
                key=f"{cache_prefix}_image_upload_{region_idx}_{manuscript_name}",
                help="Upload a screenshot of the equation or table"
            )

            if uploaded_file:
                col_img, col_ocr = st.columns([1, 2])

                with col_img:
                    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

                with col_ocr:
                    if st.button(
                        "🔍 Extract Text (OCR)",
                        key=f"{cache_prefix}_ocr_{region_idx}_{manuscript_name}"
                    ):
                        with st.spinner("Running OCR..."):
                            try:
                                ocr_text = _extract_text_from_image(uploaded_file.read())
                                fix["content"] = ocr_text
                                st.success("✅ Text extracted!")
                                st.code(ocr_text)
                            except Exception as e:
                                st.error(f"OCR failed: {e}")
                                st.caption("You can still paste text manually")

        elif fix["action"] == "skip":
            st.info("⏭️ Keeping original extraction for this region")


def _extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from image using OCR.

    Args:
        image_bytes: Image file bytes

    Returns:
        Extracted text
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        # Open image
        image = Image.open(io.BytesIO(image_bytes))

        # Run OCR
        text = pytesseract.image_to_string(image)

        return text.strip()

    except ImportError:
        raise Exception(
            "OCR requires pytesseract. Install with: pip install pytesseract\n"
            "Also requires tesseract binary: https://github.com/tesseract-ocr/tesseract"
        )


def _apply_fixes(text: str, regions: List[MathRegion], fixes: Dict) -> str:
    """
    Apply user fixes to text.

    Args:
        text: Original text
        regions: List of detected regions
        fixes: Dictionary of fixes keyed by region_id

    Returns:
        Modified text with fixes applied
    """
    # Sort regions by position (reverse order for index preservation)
    regions_with_fixes = []
    for i, region in enumerate(regions, 1):
        region_id = f"region_{i}"
        fix = fixes.get(region_id, {})

        if fix.get("action") in ["llm", "manual", "image"] and fix.get("content"):
            regions_with_fixes.append((region, fix["content"]))

    # Sort by position (backwards to preserve indices)
    regions_with_fixes.sort(key=lambda x: x[0].start_idx, reverse=True)

    # Apply fixes
    for region, new_content in regions_with_fixes:
        text = text[:region.start_idx] + new_content + text[region.end_idx:]

    return text
