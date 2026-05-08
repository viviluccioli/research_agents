#!/usr/bin/env python3
"""
run_section_eval.py

CLI wrapper for section evaluator - called by PHP
Evaluates paper sections and returns JSON results
"""
import sys
import json
import argparse
from pathlib import Path

# Add app_system to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from section_eval.evaluator import SectionEvaluator
    from section_eval.text_extraction import decode_file
    from section_eval.section_detection import detect_sections, extract_sections_from_text
    from section_eval.scoring import compute_overall_score
except ImportError as e:
    print(json.dumps({'error': f'Failed to import required modules: {e}'}), file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Run section evaluator')
    parser.add_argument('--file', required=True, help='Path to paper file')
    parser.add_argument('--paper-type', required=True, help='Paper type (empirical, theoretical, policy, etc.)')
    parser.add_argument('--mode', default='auto', choices=['auto', 'manual'], help='Detection mode')
    parser.add_argument('--sections', default=None, help='JSON dict of section_name: section_text for manual mode')
    parser.add_argument('--paper-context', default=None, help='Optional paper context/summary')

    args = parser.parse_args()

    # Read file
    file_path = Path(args.file)
    if not file_path.exists():
        print(json.dumps({'error': f'File not found: {args.file}'}), file=sys.stderr)
        sys.exit(1)

    try:
        evaluator = SectionEvaluator()

        if args.mode == 'auto':
            # Auto-detect workflow
            file_content = file_path.read_bytes()
            filename = file_path.name

            # Extract text
            text = decode_file(filename, file_content, warn_fn=lambda x: None)

            # Detect sections
            detected = detect_sections(text, llm=None)  # Uses default LLM from utils.cm

            # Extract section texts
            section_texts = extract_sections_from_text(text, detected, desired_sections=None)

        else:
            # Manual mode - sections provided as JSON
            if not args.sections:
                print(json.dumps({'error': 'Manual mode requires --sections parameter'}), file=sys.stderr)
                sys.exit(1)

            try:
                section_texts = json.loads(args.sections)
            except json.JSONDecodeError as e:
                print(json.dumps({'error': f'Invalid JSON for sections: {e}'}), file=sys.stderr)
                sys.exit(1)

        # Evaluate each section
        results = {}
        paper_context = args.paper_context if args.paper_context else ''

        for section_name, section_text in section_texts.items():
            try:
                evaluation = evaluator.evaluate_section(
                    section_name=section_name,
                    section_text=section_text,
                    paper_type=args.paper_type,
                    paper_context=paper_context,
                    figures_external=False
                )
                results[section_name] = evaluation
            except Exception as e:
                # Include section-specific errors in results
                results[section_name] = {
                    'error': str(e),
                    'section_name': section_name
                }

        # Compute overall score
        section_scores = {}
        for name, eval_result in results.items():
            if 'section_score' in eval_result:
                section_scores[name] = eval_result['section_score']

        if section_scores:
            overall = compute_overall_score(section_scores, args.paper_type)
        else:
            overall = {
                'error': 'No sections were successfully evaluated',
                'overall_score': 0,
                'publication_readiness': 'Unable to determine'
            }

        # Output complete results
        output = {
            'sections': results,
            'overall': overall,
            'paper_type': args.paper_type,
            'mode': args.mode,
            'num_sections': len(section_texts)
        }

        print(json.dumps(output, indent=2, default=str))

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
