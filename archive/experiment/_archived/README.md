# Archived Experiment Scripts

This directory contains one-off analysis and testing scripts used during experiment development. These are **not part of the core batch processing pipeline** but are preserved for reference.

## Analysis Scripts (Post-hoc Data Analysis)

- **`analyze_existing_results.py`** — Retrospective analysis of batch results CSV
- **`analyze_personas.py`** — Persona selection frequency analysis
- **`analyze_trajectories.py`** — Verdict trajectory analysis (depends on analyze_existing_results)
- **`analyze_variance.py`** — Variance analysis by tier
- **`add_calibration_metrics.py`** — Adds calibration metrics to existing CSV
- **`analyze_calibration.py`** — Calibration analysis
- **`enhance_existing_csv.py`** — Enhances CSV with additional computed metrics

## Test Scripts (Development/Debugging)

- **`test_scoring_system.py`** — Tests scoring system logic
- **`run_scoring_test.sh`** — Wrapper for scoring tests
- **`test_single_paper_scoring.sh`** — Single-paper scoring test

## Status

These scripts were used for specific analyses during development and are no longer actively maintained. Use at your own risk.

For current batch processing, use the scripts in the parent directory:
- `batch_referee_reports.py`
- `run_experiment.sh`
- `test_setup.py`
