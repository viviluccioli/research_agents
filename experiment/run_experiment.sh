#!/bin/bash
#
# Helper script to run batch referee report experiment
#
# Usage:
#   bash run_experiment.sh [limit]
#
# Example:
#   bash run_experiment.sh        # Run on all PDFs
#   bash run_experiment.sh 5      # Run on first 5 PDFs (for testing)
#

set -e  # Exit on error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Default paths (modify these for your setup)
PDF_DIR="${REPO_ROOT}/experiment-papers/IFDP_2020/pdfs"
GROUND_TRUTH="${REPO_ROOT}/experiment-papers/IFDP_2020/IFDP_2020_tracking_clean.csv"
OUTPUT_DIR="${SCRIPT_DIR}/results"

# Optional limit argument
LIMIT="${1:-}"

echo "================================================================================"
echo "Batch Referee Report Experiment"
echo "================================================================================"
echo "PDF Directory: ${PDF_DIR}"
echo "Ground Truth:  ${GROUND_TRUTH}"
echo "Output Dir:    ${OUTPUT_DIR}"
if [ -n "$LIMIT" ]; then
    echo "Limit:         ${LIMIT} PDFs"
fi
echo "================================================================================"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Virtual environment not detected. Attempting to activate..."
    if [ -f "${REPO_ROOT}/venv/bin/activate" ]; then
        source "${REPO_ROOT}/venv/bin/activate"
        echo "Virtual environment activated: ${VIRTUAL_ENV}"
    else
        echo "Error: Virtual environment not found at ${REPO_ROOT}/venv"
        echo "Please run: python -m venv ${REPO_ROOT}/venv && source ${REPO_ROOT}/venv/bin/activate"
        exit 1
    fi
fi

# Check if paths exist
if [ ! -d "$PDF_DIR" ]; then
    echo "Error: PDF directory not found: ${PDF_DIR}"
    echo "Please update PDF_DIR in this script or create the directory."
    exit 1
fi

if [ ! -f "$GROUND_TRUTH" ]; then
    echo "Error: Ground truth CSV not found: ${GROUND_TRUTH}"
    echo "Please update GROUND_TRUTH in this script or create the file."
    exit 1
fi

# Build command
CMD="python3 ${SCRIPT_DIR}/batch_referee_reports.py --pdf-dir \"${PDF_DIR}\" --ground-truth \"${GROUND_TRUTH}\" --output-dir \"${OUTPUT_DIR}\""

if [ -n "$LIMIT" ]; then
    CMD="${CMD} --limit ${LIMIT}"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Run the experiment
eval $CMD

echo ""
echo "================================================================================"
echo "Experiment complete! Results saved to: ${OUTPUT_DIR}"
echo "================================================================================"
