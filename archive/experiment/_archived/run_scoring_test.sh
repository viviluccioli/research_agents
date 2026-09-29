#!/bin/bash
# Run scoring system test on 5 sample papers

echo "============================================================"
echo "Running Hybrid Scoring Test on 5 Papers"
echo "============================================================"

cd /casl/home/m1vcl00/FS-CASL/research_agents/experiment

python3 batch_referee_reports.py \
    --pdf-dir ifdp_sample/ \
    --ground-truth ../experiment-papers/IFDP_2020/IFDP_2020_tracking_clean.csv \
    --output-dir results/ \
    --limit 5

echo ""
echo "============================================================"
echo "Test Complete!"
echo "============================================================"
echo "Results saved to: results/referee_batch_results_TIMESTAMP.csv"
echo ""
echo "Check columns:"
echo "  - persona_X_round1_score (R1 quality scores)"
echo "  - persona_X_final_score (R2C quality scores)"
echo "  - round2c_consensus_score_numeric (weighted average)"
echo ""
echo "Compare with:"
echo "  - round2c_consensus_score (categorical method)"
echo ""
