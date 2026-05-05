#!/bin/bash
# Test scoring system on a single paper to validate fixes

echo "============================================================"
echo "Testing Score Extraction on Single Paper"
echo "============================================================"

cd /casl/home/m1vcl00/FS-CASL/research_agents/experiment

echo ""
echo "Running referee system on ifdp-2020-4.pdf (should take ~3-4 min)..."
echo ""

python3 batch_referee_reports.py \
    --pdf-dir ifdp_sample/ \
    --ground-truth ../experiment-papers/IFDP_2020/IFDP_2020_tracking_clean.csv \
    --output-dir results/ \
    --limit 1

echo ""
echo "============================================================"
echo "Checking Score Extraction..."
echo "============================================================"

# Get the most recent CSV
LATEST_CSV=$(ls -t results/referee_batch_results_*.csv | head -1)

echo ""
echo "Analyzing: $LATEST_CSV"
echo ""

python3 << 'EOF'
import pandas as pd
import glob

csv_files = glob.glob('results/referee_batch_results_*.csv')
latest = max(csv_files, key=lambda x: x.split('_')[-1])

df = pd.read_csv(latest)
paper = df.iloc[0]

print("="*60)
print("SCORE EXTRACTION TEST RESULTS")
print("="*60)

personas = [
    (1, paper.get('persona_1_name')),
    (2, paper.get('persona_2_name')),
    (3, paper.get('persona_3_name'))
]

r1_scores = []
r2c_scores = []

for num, name in personas:
    if name:
        r1_score = paper.get(f'persona_{num}_round1_score')
        r2c_score = paper.get(f'persona_{num}_final_score')

        r1_status = "✓ PASS" if pd.notna(r1_score) else "✗ FAIL"
        r2c_status = "✓ PASS" if pd.notna(r2c_score) else "✗ FAIL"

        print(f"\nPersona {num} ({name}):")
        print(f"  R1 score:  {r1_status} (value: {r1_score})")
        print(f"  R2C score: {r2c_status} (value: {r2c_score})")

        if pd.notna(r1_score):
            r1_scores.append(r1_score)
        if pd.notna(r2c_score):
            r2c_scores.append(r2c_score)

print("\n" + "="*60)
print("OVERALL RESULTS")
print("="*60)

r1_captured = len(r1_scores)
r2c_captured = len(r2c_scores)
total_personas = len([n for _, n in personas if n])

print(f"R1 scores captured:  {r1_captured}/{total_personas}")
print(f"R2C scores captured: {r2c_captured}/{total_personas}")

# Check numeric consensus
r1_numeric = paper.get('round1_consensus_score_numeric')
r2c_numeric = paper.get('round2c_consensus_score_numeric')

print(f"\nR1 numeric consensus:  {r1_numeric if pd.notna(r1_numeric) else 'None'}")
print(f"R2C numeric consensus: {r2c_numeric if pd.notna(r2c_numeric) else 'None'}")

print("\n" + "="*60)
if r1_captured == total_personas and r2c_captured == total_personas:
    print("✓ SUCCESS: All scores captured!")
    print("System is ready for full batch run.")
elif r2c_captured == 0:
    print("✗ FAIL: R2C scores still not captured.")
    print("Prompts may need further adjustment.")
    print("Check persona reports to see format used.")
else:
    print("⚠ PARTIAL: Some scores captured, some missing.")
    print(f"R1: {r1_captured}/{total_personas}, R2C: {r2c_captured}/{total_personas}")
print("="*60)
EOF

echo ""
echo "Test complete!"
echo ""
