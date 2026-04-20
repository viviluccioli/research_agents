"""
PyMuPDF-based PDF extractor with figure and table extraction.

This module provides advanced PDF extraction using PyMuPDF (fitz), supporting:
- Text extraction with better handling of multi-column layouts
- Figure extraction (embedded images and rendered vector graphics)
- Table extraction with OCR
- Figure caption parsing and multi-panel detection
- Metadata extraction

Falls back to pdfplumber if PyMuPDF is not available or fails.
"""
import re
import tempfile
import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from io import BytesIO
import logging

# Try to import PyMuPDF dependencies
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Try to import OCR dependencies (optional)
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Always available fallback
import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class Figure:
    """Represents an extracted figure from a PDF."""
    figure_number: Optional[str] = None  # "1", "3a", extracted from caption
    figure_id: str = ""  # Normalized ID: "figure_1", "figure_3a"
    page_number: int = 0  # 1-indexed page number
    image_data: bytes = b""  # PNG/JPEG bytes
    image_format: str = "png"  # "png", "jpeg"
    caption: Optional[str] = None  # Full caption text
    title: Optional[str] = None  # Figure title (first sentence)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # (x0, y0, x1, y1)
    width_px: int = 0  # Width in pixels
    height_px: int = 0  # Height in pixels
    dpi: Optional[float] = None  # Estimated DPI
    references_in_text: List[str] = field(default_factory=list)  # Text snippets referencing figure
    is_multi_panel: bool = False  # Has panels (a, b, c)?
    panels: Optional[List[str]] = None  # ["a", "b", "c"]


@dataclass
class ExtractedContent:
    """Container for all extracted PDF content."""
    text: str
    figures: List[Figure] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extractor_used: str = "unknown"  # "pymupdf", "pdfplumber"


# ==================== FIGURE EXTRACTION UTILITIES ====================

def find_figure_captions(page: 'fitz.Page') -> List[Dict[str, Any]]:
    """
    Find all figure captions on a page.

    Args:
        page: PyMuPDF Page object

    Returns:
        List of dicts with: figure_number, caption_text, bbox
    """
    captions = []
    blocks = page.get_text("dict")["blocks"]

    for block_idx, block in enumerate(blocks):
        if "lines" not in block:
            continue

        for line_idx, line in enumerate(block["lines"]):
            # Combine spans in this line
            line_text = " ".join([span["text"] for span in line["spans"]])

            # Match "Figure X:" or "Fig. X:" patterns
            match = re.match(
                r'^(Figure|Fig\.?)\s+(\d+[a-zA-Z]?)[\.:]\s*(.+)',
                line_text,
                re.IGNORECASE
            )

            if match:
                fig_num = match.group(2)
                caption_start = match.group(3)
                caption_start_bbox = line["bbox"]

                # Collect multi-line caption
                caption_lines = [caption_start]

                # Remaining lines in current block
                for next_line in block["lines"][line_idx + 1:]:
                    next_text = " ".join([span["text"] for span in next_line["spans"]]).strip()
                    if not next_text:
                        break
                    if re.match(r'^(Figure|Fig\.?)\s+\d+[a-zA-Z]?[\.:]\s*', next_text, re.IGNORECASE):
                        break
                    if next_text.isupper() and len(next_text) < 50:
                        break
                    caption_lines.append(next_text)

                full_caption = f"Figure {fig_num}: {' '.join(caption_lines)}"
                bbox = block["bbox"] if len(caption_lines) > 1 else caption_start_bbox

                captions.append({
                    'figure_number': fig_num,
                    'caption_text': full_caption,
                    'bbox': bbox,
                })

    return captions


def estimate_figure_bbox_from_caption(
    caption_bbox: Tuple,
    page_rect: 'fitz.Rect',
    figure_height_estimate: float = 250.0
) -> 'fitz.Rect':
    """
    Estimate figure bounding box based on caption location.

    Figures are typically above their captions in academic papers.
    """
    x0, y0, x1, y1 = caption_bbox
    margin_x = 20
    fig_x0 = max(0, x0 - margin_x)
    fig_x1 = min(page_rect.width, x1 + margin_x)
    fig_y1 = y0 - 10  # Gap between figure and caption
    fig_y0 = max(0, fig_y1 - figure_height_estimate)
    return fitz.Rect(fig_x0, fig_y0, fig_x1, fig_y1)


def normalize_figure_id(figure_number: Optional[str]) -> str:
    """
    Convert figure number to normalized ID.

    Examples: "1" -> "figure_1", "3a" -> "figure_3a"
    """
    if not figure_number:
        return "figure_unknown"
    normalized = figure_number.lower().replace(" ", "_").replace(".", "")
    return f"figure_{normalized}"


def detect_multi_panel(caption_text: str) -> Tuple[bool, Optional[List[str]]]:
    """
    Detect if figure has multiple panels from caption text.

    Looks for patterns: (a), (b), (c), "panel a", "Panel A", "subfigure a"
    """
    if not caption_text:
        return False, None

    panel_pattern = r'\(([a-h])\)|panel\s+([a-h])|subfigure\s+([a-h])'
    matches = re.findall(panel_pattern, caption_text.lower())

    if not matches:
        return False, None

    panels = []
    for match_tuple in matches:
        panel_letter = next((g for g in match_tuple if g), None)
        if panel_letter and panel_letter not in panels:
            panels.append(panel_letter)

    if len(panels) >= 2:
        return True, sorted(panels)

    return False, None


def extract_title_from_caption(caption_text: str) -> Optional[str]:
    """
    Extract figure title (first sentence) from caption.

    Example: "Figure 1: Treatment effects over time. Details..."
             -> "Treatment effects over time"
    """
    if not caption_text:
        return None

    caption = re.sub(r'^Figure\s+\d+[a-z]*:\s*', '', caption_text, flags=re.IGNORECASE)
    match = re.match(r'^([^.!?]+)[.!?]?', caption)
    if match:
        title = match.group(1).strip()
        if len(title) > 100:
            title = title[:97] + "..."
        return title

    return caption[:100] if len(caption) > 100 else caption


def find_text_references(figure_number: str, text: str) -> List[str]:
    """
    Find all text snippets that reference a specific figure.

    Returns list of text snippets (limited to first 5).
    """
    references = []
    pattern = rf'[^.]*\b(Figure|Fig\.?)\s+{re.escape(figure_number)}\b[^.]*\.'
    matches = re.finditer(pattern, text, re.IGNORECASE)

    for match in matches:
        snippet = match.group(0).strip()
        if snippet and snippet not in references:
            references.append(snippet)

    return references[:5]


def render_figure_region(
    page: 'fitz.Page',
    bbox: 'fitz.Rect',
    resolution_scale: float = 2.0
) -> Tuple[bytes, int, int]:
    """
    Render a specific region of a PDF page as PNG.

    Args:
        page: PyMuPDF Page
        bbox: Bounding box region
        resolution_scale: Scale factor (2.0 = ~150 DPI)

    Returns:
        (image_bytes, width_px, height_px)
    """
    mat = fitz.Matrix(resolution_scale, resolution_scale)
    pix = page.get_pixmap(matrix=mat, clip=bbox)
    image_bytes = pix.tobytes("png")
    return image_bytes, pix.width, pix.height


# ==================== TABLE EXTRACTION WITH OCR ====================

def is_table_caption(caption: str) -> bool:
    """Check if caption indicates a table (not a figure)."""
    if not caption:
        return False
    return bool(re.search(r'\bTable\s+\d+', caption, re.IGNORECASE))


def is_table_by_aspect_ratio(width: int, height: int) -> bool:
    """
    Heuristic: tables often have unusual aspect ratios.

    Very wide (>2.5:1) or very tall (<0.4:1) suggests a table.
    """
    aspect_ratio = width / height if height > 0 else 0
    return aspect_ratio > 2.5 or aspect_ratio < 0.4


def extract_text_from_table_image(image_data: bytes, image_format: str = "png") -> Tuple[str, float]:
    """
    Extract text from table image using OCR.

    Returns:
        (extracted_text, confidence_score)
    """
    if not OCR_AVAILABLE:
        return "[OCR not available - install pytesseract and Pillow]", 0.0

    try:
        img = Image.open(BytesIO(image_data))
        img = img.convert('L')  # Grayscale for better OCR

        # PSM 6 = uniform block of text (good for tables)
        config = '--psm 6'
        text = pytesseract.image_to_string(img, config=config)

        # Try to get confidence data
        try:
            data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
            confidences = [conf for conf in data['conf'] if conf != -1]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        except:
            avg_confidence = 0.5  # Default moderate confidence

        return text.strip(), avg_confidence / 100.0

    except Exception as e:
        logger.warning(f"OCR error: {e}")
        return f"[OCR Error: {str(e)}]", 0.0


def format_as_markdown_table(ocr_text: str, confidence: float) -> str:
    """
    Format OCR text as markdown table.

    Best-effort formatting. Adds warning for low-confidence extractions.
    """
    if not ocr_text or ocr_text.startswith("[OCR"):
        return ocr_text

    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]

    if not lines:
        return ocr_text

    # Simple heuristic: check for multi-column structure
    if len(lines) >= 2:
        first_tokens = re.split(r'\s{2,}|\t', lines[0])

        if len(first_tokens) >= 2:
            markdown_lines = []
            markdown_lines.append('| ' + ' | '.join(first_tokens) + ' |')
            markdown_lines.append('| ' + ' | '.join(['---'] * len(first_tokens)) + ' |')

            for line in lines[1:]:
                tokens = re.split(r'\s{2,}|\t', line)
                while len(tokens) < len(first_tokens):
                    tokens.append('')
                markdown_lines.append('| ' + ' | '.join(tokens[:len(first_tokens)]) + ' |')

            result = '\n'.join(markdown_lines)

            # Add warning for low confidence
            if confidence < 0.5:
                result = f"⚠️ *Low confidence OCR ({confidence:.0%})*\n\n{result}"

            return result

    # Fallback: code block
    result = f"```\n{ocr_text}\n```"
    if confidence < 0.5:
        result = f"⚠️ *Low confidence OCR ({confidence:.0%})*\n\n{result}"
    return result


def find_table_number(caption: str) -> Optional[str]:
    """Extract table number from caption."""
    if not caption:
        return None
    match = re.search(r'Table\s+(\d+[a-zA-Z]?)', caption, re.IGNORECASE)
    return match.group(1) if match else None


# ==================== MULTI-COLUMN LAYOUT DETECTION ====================

def detect_multi_column_layout(page: 'fitz.Page') -> bool:
    """
    Detect if a page has multi-column layout.

    Uses heuristic: check if text blocks are clustered in distinct horizontal regions.
    """
    blocks = page.get_text("dict")["blocks"]

    # Get x-coordinates of all text blocks
    x_coords = []
    for block in blocks:
        if "lines" in block:
            bbox = block["bbox"]
            x_center = (bbox[0] + bbox[2]) / 2
            x_coords.append(x_center)

    if len(x_coords) < 10:  # Not enough blocks to determine
        return False

    # Simple clustering: check if blocks split into left/right halves
    x_coords_sorted = sorted(x_coords)
    page_width = page.rect.width
    midpoint = page_width / 2

    left_blocks = [x for x in x_coords if x < midpoint * 0.8]
    right_blocks = [x for x in x_coords if x > midpoint * 1.2]

    # If significant blocks on both sides, likely multi-column
    return len(left_blocks) > 5 and len(right_blocks) > 5


# ==================== MAIN EXTRACTION FUNCTIONS ====================

def extract_text_and_figures(
    pdf_path: str,
    min_figure_size: int = 100,
    resolution_scale: float = 2.0,
    extract_tables: bool = True
) -> ExtractedContent:
    """
    Extract text and figures from PDF using PyMuPDF.

    Args:
        pdf_path: Path to PDF file
        min_figure_size: Minimum dimension (px) for extracted images
        resolution_scale: Rendering resolution (2.0 = ~150 DPI)
        extract_tables: Extract tables via OCR

    Returns:
        ExtractedContent with text, figures, tables, and metadata
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF (fitz) not available. Install with: pip install PyMuPDF>=1.23.0")

    try:
        doc = fitz.open(pdf_path)

        paper_text_pages = []
        figures = []
        tables = []
        multi_column_pages = []

        for page_num, page in enumerate(doc):
            # Detect multi-column layout
            is_multi_col = detect_multi_column_layout(page)
            if is_multi_col:
                multi_column_pages.append(page_num + 1)

            # Extract text
            text = page.get_text("text")
            if text.strip():
                paper_text_pages.append(text)

            # Extract embedded images
            embedded_imgs, embedded_tables = _extract_embedded_images(
                page, doc, page_num + 1, min_figure_size, extract_tables
            )
            figures.extend(embedded_imgs)
            tables.extend(embedded_tables)

            # Extract caption-based figures (vector graphics)
            caption_figs = _extract_caption_based_figures(
                page, page_num + 1, resolution_scale
            )
            figures.extend(caption_figs)

        doc.close()

        # Combine text
        paper_text = "\n\n".join(paper_text_pages)

        # Insert tables into paper text
        if extract_tables:
            for table_info in tables:
                table_num = table_info['table_number']
                table_md = table_info['markdown']
                caption = table_info.get('caption')
                confidence = table_info.get('confidence', 1.0)

                paper_text = _insert_table_into_text(paper_text, table_num, table_md, caption, confidence)

        # Post-process: find text references for figures
        for figure in figures:
            if figure.figure_number:
                figure.references_in_text = find_text_references(figure.figure_number, paper_text)

        # Deduplicate figures
        figures = _deduplicate_figures(figures)

        # Build metadata
        metadata = {
            'total_pages': len(paper_text_pages),
            'total_figures': len(figures),
            'total_tables': len(tables),
            'multi_column_pages': multi_column_pages,
            'has_multi_column': len(multi_column_pages) > 0,
        }

        return ExtractedContent(
            text=paper_text,
            figures=figures,
            tables=tables,
            metadata=metadata,
            extractor_used="pymupdf"
        )

    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        raise


def _extract_embedded_images(
    page: 'fitz.Page',
    doc: 'fitz.Document',
    page_num: int,
    min_size: int,
    extract_tables: bool = True
) -> Tuple[List[Figure], List[Dict[str, Any]]]:
    """
    Extract embedded raster images from page.

    Returns:
        (figures, tables)
    """
    figures = []
    tables = []
    image_list = page.get_images(full=True)

    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = doc.extract_image(xref)

        # Skip if extraction failed
        if not base_image or base_image.get("image") is None:
            logger.debug(f"Skipping image {img_index} on page {page_num}: extraction returned None")
            continue

        # Filter small images (logos, icons)
        if base_image["width"] < min_size or base_image["height"] < min_size:
            continue

        # Get image position
        img_rects = page.get_image_rects(xref)
        bbox = tuple(img_rects[0]) if img_rects else (0, 0, base_image["width"], base_image["height"])

        # Try to find nearby caption
        caption = _find_nearby_caption(page, bbox)
        figure_number = _extract_figure_number(caption) if caption else None

        # Check if this is a table
        is_table = False
        if extract_tables and caption:
            is_table = is_table_caption(caption)

        if not is_table and extract_tables:
            is_table = is_table_by_aspect_ratio(base_image["width"], base_image["height"])

        if is_table:
            # Extract table via OCR
            table_number = find_table_number(caption) if caption else f"unknown_{page_num}_{img_index}"
            ocr_text, confidence = extract_text_from_table_image(base_image["image"], base_image["ext"])
            table_markdown = format_as_markdown_table(ocr_text, confidence)

            tables.append({
                'table_number': table_number,
                'markdown': table_markdown,
                'caption': caption,
                'confidence': confidence,
                'page': page_num
            })
            continue

        # It's a figure
        figure_id = normalize_figure_id(figure_number) if figure_number else f"figure_page{page_num}_img{img_index}"
        is_multi_panel, panels = detect_multi_panel(caption or "")
        title = extract_title_from_caption(caption) if caption else None

        figures.append(Figure(
            figure_number=figure_number,
            figure_id=figure_id,
            page_number=page_num,
            image_data=base_image["image"],
            image_format=base_image["ext"],
            caption=caption,
            title=title,
            bbox=bbox,
            width_px=base_image["width"],
            height_px=base_image["height"],
            dpi=base_image.get("xres"),
            references_in_text=[],
            is_multi_panel=is_multi_panel,
            panels=panels
        ))

    return figures, tables


def _extract_caption_based_figures(
    page: 'fitz.Page',
    page_num: int,
    resolution_scale: float
) -> List[Figure]:
    """
    Find figures by detecting captions and rendering the region above.

    Handles vector graphics figures in LaTeX PDFs.
    """
    figures = []
    captions = find_figure_captions(page)

    for cap_info in captions:
        figure_number = cap_info['figure_number']
        caption_text = cap_info['caption_text']
        caption_bbox = cap_info['bbox']

        # Estimate figure bbox (above caption)
        figure_bbox = estimate_figure_bbox_from_caption(caption_bbox, page.rect)

        # Render region as image
        image_bytes, width, height = render_figure_region(page, figure_bbox, resolution_scale)

        figure_id = normalize_figure_id(figure_number)
        is_multi_panel, panels = detect_multi_panel(caption_text)
        title = extract_title_from_caption(caption_text)

        figures.append(Figure(
            figure_number=figure_number,
            figure_id=figure_id,
            page_number=page_num,
            image_data=image_bytes,
            image_format="png",
            caption=caption_text,
            title=title,
            bbox=tuple(figure_bbox),
            width_px=width,
            height_px=height,
            dpi=72.0 * resolution_scale,
            references_in_text=[],
            is_multi_panel=is_multi_panel,
            panels=panels
        ))

    return figures


def _find_nearby_caption(page: 'fitz.Page', bbox: Tuple) -> Optional[str]:
    """Find caption text near an image bounding box."""
    blocks = page.get_text("dict")["blocks"]
    img_y_center = (bbox[1] + bbox[3]) / 2

    for block in blocks:
        if "lines" not in block:
            continue

        block_bbox = block["bbox"]
        block_y_center = (block_bbox[1] + block_bbox[3]) / 2

        # Check if block is near image (within ~100 points vertically)
        if abs(block_y_center - img_y_center) < 100:
            text = " ".join([
                " ".join([span["text"] for span in line["spans"]])
                for line in block["lines"]
            ])

            if re.search(r'\bFigure\s+\d+', text, re.IGNORECASE):
                return text

    return None


def _extract_figure_number(caption: str) -> Optional[str]:
    """Extract figure number from caption text."""
    match = re.search(r'Figure\s+(\d+[a-zA-Z]?)', caption, re.IGNORECASE)
    return match.group(1) if match else None


def _insert_table_into_text(
    paper_text: str,
    table_number: str,
    table_markdown: str,
    caption: Optional[str],
    confidence: float
) -> str:
    """Insert extracted table into paper text at appropriate location."""
    # Find where "Table X" is mentioned
    pattern = rf'\bTable\s+{re.escape(table_number)}\b'
    match = re.search(pattern, paper_text, re.IGNORECASE)

    # Build insertion text
    insertion = f"\n\n[TABLE {table_number} - EXTRACTED VIA OCR"
    if confidence < 0.7:
        insertion += f" - CONFIDENCE: {confidence:.0%}"
    insertion += "]\n"

    if caption:
        insertion += f"{caption}\n\n"
    insertion += f"{table_markdown}\n"

    if not match:
        # Append at end
        return paper_text + insertion

    # Insert after match
    insert_pos = match.end()
    return paper_text[:insert_pos] + insertion + paper_text[insert_pos:]


def _deduplicate_figures(figures: List[Figure]) -> List[Figure]:
    """
    Remove duplicate figures based on:
    1. Same figure_number + page_number
    2. Overlapping bboxes on same page
    """
    # First pass: deduplicate by figure_number + page
    seen_by_number = {}

    for fig in figures:
        key = (fig.figure_number, fig.page_number)

        if key not in seen_by_number:
            seen_by_number[key] = fig
        else:
            existing = seen_by_number[key]
            # Prefer figures with captions
            if fig.caption and not existing.caption:
                seen_by_number[key] = fig
            # Prefer embedded images (JPEG) over rendered (PNG)
            elif fig.caption and existing.caption:
                if fig.image_format == "jpeg":
                    seen_by_number[key] = fig

    figures = list(seen_by_number.values())

    # Second pass: check overlapping bboxes
    final_figures = []

    for i, fig1 in enumerate(figures):
        is_duplicate = False

        for j, fig2 in enumerate(figures):
            if i >= j:
                continue

            if fig1.page_number != fig2.page_number:
                continue

            if _bboxes_overlap(fig1.bbox, fig2.bbox, threshold=0.5):
                if fig2.figure_number and fig2.caption:
                    is_duplicate = True
                    break

        if not is_duplicate:
            final_figures.append(fig1)

    return final_figures


def _bboxes_overlap(bbox1: Tuple, bbox2: Tuple, threshold: float = 0.5) -> bool:
    """Check if two bounding boxes overlap significantly."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    overlap_x_min = max(x1_min, x2_min)
    overlap_y_min = max(y1_min, y2_min)
    overlap_x_max = min(x1_max, x2_max)
    overlap_y_max = min(y1_max, y2_max)

    if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
        return False

    overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    min_area = min(area1, area2)
    overlap_ratio = overlap_area / min_area if min_area > 0 else 0

    return overlap_ratio > threshold


# ==================== FALLBACK TO PDFPLUMBER ====================

def extract_with_pdfplumber_fallback(file_content: bytes) -> ExtractedContent:
    """
    Fallback extraction using pdfplumber (text + tables only, no figures).

    Args:
        file_content: PDF file bytes

    Returns:
        ExtractedContent with text and tables (no figures)
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file.flush()
        temp_path = temp_file.name

    try:
        text = ""
        total_tables = 0

        with pdfplumber.open(temp_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                text += page_text

                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        total_tables += 1
                        text += f"\n\n[TABLE {total_tables} - Page {page_num}]\n"
                        text += _format_table_markdown_simple(table)
                        text += "\n[END TABLE]\n\n"

                text += f"\n\n--- PAGE {page_num} ---\n\n"

        metadata = {
            'total_pages': len(pdf.pages),
            'total_figures': 0,
            'total_tables': total_tables,
        }

        return ExtractedContent(
            text=text,
            figures=[],
            tables=[],
            metadata=metadata,
            extractor_used="pdfplumber"
        )

    finally:
        os.unlink(temp_path)


def _format_table_markdown_simple(table: List[List]) -> str:
    """Simple markdown table formatting for pdfplumber tables."""
    if not table or len(table) < 2:
        return ""

    markdown_lines = []

    # Header row
    header = [str(cell or "") for cell in table[0]]
    markdown_lines.append('| ' + ' | '.join(header) + ' |')

    # Separator
    markdown_lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')

    # Data rows
    for row in table[1:]:
        cells = [str(cell or "") for cell in row]
        while len(cells) < len(header):
            cells.append('')
        markdown_lines.append('| ' + ' | '.join(cells[:len(header)]) + ' |')

    return '\n'.join(markdown_lines)


# ==================== PUBLIC API ====================

def extract_pdf_with_figures(
    file_content: bytes,
    use_pymupdf: bool = True,
    min_figure_size: int = 100,
    resolution_scale: float = 2.0,
    extract_tables: bool = True
) -> ExtractedContent:
    """
    Extract text, figures, and tables from PDF.

    Args:
        file_content: PDF file bytes
        use_pymupdf: Use PyMuPDF if available (falls back to pdfplumber if False or on error)
        min_figure_size: Minimum dimension for figures (PyMuPDF only)
        resolution_scale: Rendering resolution for vector figures (PyMuPDF only)
        extract_tables: Extract tables via OCR (PyMuPDF) or structured extraction (pdfplumber)

    Returns:
        ExtractedContent with text, figures, tables, metadata
    """
    # Validate file_content
    if file_content is None:
        raise ValueError("file_content cannot be None")
    if not isinstance(file_content, bytes):
        raise TypeError(f"file_content must be bytes, got {type(file_content)}")
    if len(file_content) == 0:
        raise ValueError("file_content cannot be empty")

    if use_pymupdf and PYMUPDF_AVAILABLE:
        # Try PyMuPDF first
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            temp_path = temp_file.name

        try:
            logger.info("Attempting PDF extraction with PyMuPDF...")
            result = extract_text_and_figures(
                temp_path,
                min_figure_size=min_figure_size,
                resolution_scale=resolution_scale,
                extract_tables=extract_tables
            )
            logger.info(f"PyMuPDF extraction successful: {result.metadata['total_figures']} figures, "
                       f"{result.metadata['total_tables']} tables")
            return result

        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}. Falling back to pdfplumber...")

        finally:
            os.unlink(temp_path)

    # Fallback to pdfplumber
    logger.info("Using pdfplumber for PDF extraction (no figure extraction)")
    return extract_with_pdfplumber_fallback(file_content)
