#!/usr/bin/env python3
"""
run_referee.py

CLI wrapper for referee system - called by PHP
Runs the multi-agent debate pipeline and returns JSON results
"""
import sys
import json
import argparse
from pathlib import Path

# Add app_system to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from referee.engine import execute_debate_pipeline
    from referee._utils.pdf_extractor_v2 import extract_pdf_with_figures, PYMUPDF_AVAILABLE
    from section_eval.text_extraction import decode_file
    import asyncio
except ImportError as e:
    print(json.dumps({'error': f'Failed to import required modules: {e}'}), file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Run referee evaluation pipeline')
    parser.add_argument('--file', required=True, help='Path to paper file')
    parser.add_argument('--paper-type', default=None, help='Paper type (empirical, theoretical, policy)')
    parser.add_argument('--custom-context', default=None, help='Custom evaluation context')
    parser.add_argument('--personas', default=None, help='Comma-separated persona names for manual selection')
    parser.add_argument('--weights', default=None, help='JSON dict of persona weights')
    parser.add_argument('--use-cache', default='true', help='Use caching (true/false)')
    parser.add_argument('--enable-quote-validation', default='true', help='Enable quote validation (true/false)')
    parser.add_argument('--session-id', required=True, help='PHP session ID for progress tracking')

    args = parser.parse_args()

    # Read file
    file_path = Path(args.file)
    if not file_path.exists():
        print(json.dumps({'error': f'File not found: {args.file}'}), file=sys.stderr)
        sys.exit(1)

    try:
        # Extract text based on file type
        file_content = file_path.read_bytes()
        filename = file_path.name

        if file_path.suffix.lower() == '.pdf':
            # Use PDF extractor
            if PYMUPDF_AVAILABLE:
                extraction = extract_pdf_with_figures(file_content)
                paper_text = extraction.text
            else:
                # Fallback to decode_file
                paper_text = decode_file(filename, file_content, warn_fn=lambda x: None)
        else:
            # For .txt, .tex files
            paper_text = decode_file(filename, file_content, warn_fn=lambda x: None)

    except Exception as e:
        print(json.dumps({'error': f'Failed to extract text: {e}'}), file=sys.stderr)
        sys.exit(1)

    # Parse boolean arguments
    use_cache = args.use_cache.lower() == 'true'
    enable_quote_validation = args.enable_quote_validation.lower() == 'true'

    # Parse manual personas
    manual_personas = None
    if args.personas:
        manual_personas = [p.strip() for p in args.personas.split(',')]

    # Parse manual weights
    manual_weights = None
    if args.weights:
        try:
            manual_weights = json.loads(args.weights)
        except json.JSONDecodeError:
            print(json.dumps({'error': 'Invalid JSON for weights parameter'}), file=sys.stderr)
            sys.exit(1)

    # Progress callback - write to file for PHP to read
    def progress_callback(message, progress):
        progress_file = Path(f'/tmp/referee_progress_{args.session_id}.json')
        try:
            progress_file.write_text(json.dumps({
                'progress': progress,
                'message': message,
                'timestamp': __import__('time').time()
            }))
        except Exception:
            pass  # Ignore errors writing progress

    # Run debate pipeline
    try:
        results = asyncio.run(execute_debate_pipeline(
            paper_text=paper_text,
            progress_callback=progress_callback,
            paper_type=args.paper_type if args.paper_type else None,
            custom_context=args.custom_context if args.custom_context else None,
            manual_personas=manual_personas,
            manual_weights=manual_weights,
            enable_quote_validation=enable_quote_validation,
            use_cache=use_cache,
            force_refresh=False
        ))

        # Output results as JSON to stdout
        print(json.dumps(results, indent=2, default=str))

        # Final progress update
        progress_callback('Complete', 1.0)

    except Exception as e:
        import traceback
        error_details = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(json.dumps(error_details), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
