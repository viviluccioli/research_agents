#!/usr/bin/env python3
"""
Quick test to verify that imports and dependencies work correctly.
"""

import sys
from pathlib import Path

# Add app_system to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app_system"))

print("Testing imports...")

try:
    from referee.engine import execute_debate_pipeline, extract_verdict_from_report
    print("✓ referee.engine imports successful")
except ImportError as e:
    print(f"✗ Failed to import referee.engine: {e}")
    sys.exit(1)

try:
    from section_eval.text_extraction import extract_text_from_pdf
    print("✓ section_eval.text_extraction imports successful")
except ImportError as e:
    print(f"✗ Failed to import section_eval.text_extraction: {e}")
    sys.exit(1)

try:
    import pdfplumber
    print("✓ pdfplumber available")
except ImportError:
    print("✗ pdfplumber not installed. Run: pip install pdfplumber")
    sys.exit(1)

try:
    from config import MODEL_PRIMARY
    print(f"✓ config imports successful (MODEL_PRIMARY: {MODEL_PRIMARY})")
except ImportError as e:
    print(f"✗ Failed to import config: {e}")
    sys.exit(1)

print("\n✓ All imports successful! Ready to run batch_referee_reports.py")
