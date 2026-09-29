# -*- coding: utf-8 -*-
"""
Configuration Management for Research Agents

This module loads API credentials and configuration from environment variables
or a .env file. This keeps secrets out of version control.

Setup:
1. Copy .env.example to .env
2. Edit .env with your API credentials
3. Run the app - config is loaded automatically

Supports multiple API providers:
- OpenAI (api.openai.com)
- Anthropic (api.anthropic.com)
- Google Gemini (generativelanguage.googleapis.com)
- Custom endpoints (e.g., internal APIs)
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Try to load python-dotenv if available
try:
    from dotenv import load_dotenv

    # Load .env file from app_system directory
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded configuration from {env_path}")
    else:
        print(f"⚠️  No .env file found at {env_path}")
        print(f"   Copy .env.example to .env and add your API credentials")
except ImportError:
    print("⚠️  python-dotenv not installed. Using environment variables only.")
    print("   Install with: pip install python-dotenv")


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Get environment variable with validation.

    Args:
        key: Environment variable name
        default: Default value if not set
        required: If True, raises error if not set

    Returns:
        Value from environment or default

    Raises:
        SystemExit: If required variable is not set
    """
    value = os.environ.get(key, default)

    if required and not value:
        print(f"❌ ERROR: Required environment variable '{key}' is not set!")
        print(f"   Please set it in your .env file or environment.")
        sys.exit(1)

    if not value and default:
        print(f"⚠️  Using default value for {key}: {default}")

    return value


# =============================================================================
# API Configuration
# =============================================================================

# API Provider (for documentation purposes)
API_PROVIDER = get_env_var('API_PROVIDER', default='custom')

# API Credentials (REQUIRED)
API_KEY = get_env_var('API_KEY', required=True)
API_BASE = get_env_var('API_BASE', required=True)

# Construct full URL for chat completions endpoint
url_chat_completions = f"{API_BASE}/chat/completions"

# =============================================================================
# Model Configuration
# =============================================================================

# Primary model (used by section evaluator and all systems)
MODEL_PRIMARY = get_env_var(
    'MODEL_PRIMARY',
    default='anthropic.claude-sonnet-4-5-20250929-v1:0'
)

# Secondary model (DEPRECATED - now uses Claude 4.5 like primary)
# Kept for backwards compatibility but defaults to same as PRIMARY
MODEL_SECONDARY = get_env_var(
    'MODEL_SECONDARY',
    default='anthropic.claude-sonnet-4-5-20250929-v1:0'  # Changed from 3.7 to 4.5
)

# Tertiary model (legacy/backup - DEPRECATED)
MODEL_TERTIARY = get_env_var(
    'MODEL_TERTIARY',
    default='anthropic.claude-sonnet-4-5-20250929-v1:0'  # Changed from 3.5 to 4.5
)

# Backward compatibility aliases (for existing code)
model_selection = MODEL_PRIMARY
model_selection3 = MODEL_SECONDARY
model_selection2 = MODEL_TERTIARY

# =============================================================================
# PDF Extraction Configuration
# =============================================================================

# Use PyMuPDF for advanced PDF extraction (figures, better tables)
# Falls back to pdfplumber if False or if PyMuPDF fails
USE_PYMUPDF = get_env_var('USE_PYMUPDF', default='true').lower() in ('true', '1', 'yes')

# PyMuPDF extraction parameters
PYMUPDF_MIN_FIGURE_SIZE = int(get_env_var('PYMUPDF_MIN_FIGURE_SIZE', default='100'))
PYMUPDF_RESOLUTION_SCALE = float(get_env_var('PYMUPDF_RESOLUTION_SCALE', default='2.0'))
PYMUPDF_EXTRACT_TABLES = get_env_var('PYMUPDF_EXTRACT_TABLES', default='true').lower() in ('true', '1', 'yes')

# =============================================================================
# Validation
# =============================================================================

def validate_config():
    """Validate configuration and print summary."""
    print("\n" + "=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"API Provider:     {API_PROVIDER}")
    print(f"API Endpoint:     {API_BASE}")
    print(f"API Key:          {'*' * 20}{API_KEY[-8:] if len(API_KEY) > 8 else '****'}")
    print(f"Primary Model:    {MODEL_PRIMARY}")
    print(f"Secondary Model:  {MODEL_SECONDARY}")
    print(f"Tertiary Model:   {MODEL_TERTIARY}")
    print(f"")
    print(f"PDF Extraction:")
    print(f"  Use PyMuPDF:    {USE_PYMUPDF}")
    if USE_PYMUPDF:
        print(f"  Min Fig Size:   {PYMUPDF_MIN_FIGURE_SIZE}px")
        print(f"  Resolution:     {PYMUPDF_RESOLUTION_SCALE}x (~{int(72 * PYMUPDF_RESOLUTION_SCALE)} DPI)")
        print(f"  Extract Tables: {PYMUPDF_EXTRACT_TABLES}")
    print("=" * 80 + "\n")


# Run validation if this module is executed directly
if __name__ == "__main__":
    validate_config()
