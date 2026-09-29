#!/bin/bash
# Run the Experiment 4 app with file watcher disabled to avoid inotify limits

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

echo "Starting Experiment 4 app (10 personas)..."
echo "Access at: http://localhost:8501"
echo ""

streamlit run app_exp_4.py --server.fileWatcherType none
