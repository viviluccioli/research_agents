# Cross-Reference Deduplication for Referee Reports

**Added:** 2026-04-15
**Status:** Production-ready
**Location:** `referee/_utils/deduplicator.py`

## Overview

The deduplication system (Round 2.5) identifies and merges duplicate findings across persona reports to reduce redundancy while preserving unique perspectives. It runs automatically between Round 2C (Final Amendments) and Round 3 (Editor Decision).

## How It Works

### 1. Finding Extraction

The system automatically extracts individual findings from each persona's Round 2C report using multiple strategies:

- **Bullet points** with severity markers (`[FATAL]`, `[MAJOR]`, etc.)
- **Numbered lists** (1., 2., 3., etc.)
- **Paragraphs** containing issue indicators (problem, concern, weakness, flaw, etc.)

Each finding is enriched with metadata:
- **Severity:** Fatal, Major, Moderate, Minor (auto-detected from text)
- **Category:** Identification, Data, Methodology, Theory, Literature, etc.
- **Quotes:** Extracted quoted text from the paper
- **Keywords:** Technical terms (regression, endogeneity, instrument, etc.)

### 2. Similarity Calculation

Findings are compared using multiple metrics:

**Without sentence-transformers (keyword-based):**
- 45% Keyword overlap (technical terms, econometric concepts)
- 30% Category similarity (same issue type)
- 25% Quote overlap (citing same paper sections)

**With sentence-transformers (semantic + keywords):**
- 40% Semantic similarity (meaning-based, using embeddings)
- 25% Keyword overlap
- 20% Quote overlap
- 15% Category similarity

**Notes:**
- Findings from the same persona are never merged
- Similarity threshold is configurable (default: 0.8 = 80% similar)

### 3. Clustering

Similar findings are grouped using a greedy clustering algorithm:

1. Calculate pairwise similarities for all findings
2. Sort by similarity (highest first)
3. Merge findings exceeding the threshold
4. Respect severity differences (won't merge fatal with minor issues if preserve_distinct=true)

### 4. Merging Strategy

Each cluster is consolidated into a single finding:

```
**[Identified by: Theorist, Empiricist]**

[Representative finding text - from highest severity persona]

**Additional Perspectives:**
- Theorist: [Key insight from this persona]
- Empiricist: [Key insight from this persona]
```

**Preservation rules:**
- Keep highest severity as representative
- Combine all unique evidence
- Attribute to all contributing personas
- Preserve all unique quotes

## Configuration

Add to your `.env` file:

```bash
# Enable/disable deduplication (default: true)
ENABLE_DEDUPLICATION=true

# Similarity threshold: 0.0 (merge all) to 1.0 (merge only identical)
# Default: 0.8 (80% similar)
DEDUP_SIMILARITY_THRESHOLD=0.8

# Preserve distinct perspectives: don't merge if severity differs by 2+ levels
# (e.g., don't merge Fatal with Moderate)
# Default: true
DEDUP_PRESERVE_DISTINCT_PERSPECTIVES=true
```

## Installation

### Required Dependencies

```bash
pip install thefuzz python-Levenshtein
```

### Optional: Semantic Similarity

For improved semantic matching, install sentence-transformers:

```bash
pip install sentence-transformers
```

**Note:** This downloads a ~500MB model on first use. The system automatically falls back to keyword-based matching if not installed.

## Usage Examples

### Basic Usage (Automatic)

Deduplication runs automatically in the referee workflow. No code changes needed.

```python
from referee.workflow import RefereeWorkflow

workflow = RefereeWorkflow()
# Deduplication happens automatically during evaluation
results = workflow.run_evaluation(paper_text)
```

### Programmatic Usage

```python
from referee._utils.deduplicator import deduplicate_findings

# Persona reports from Round 2C
reports = {
    'Empiricist': "... full report text ...",
    'Theorist': "... full report text ...",
    'Historian': "... full report text ..."
}

# Run deduplication
result = deduplicate_findings(
    reports=reports,
    paper_text=paper_text,
    similarity_threshold=0.8,
    preserve_distinct=True
)

# Access results
stats = result['statistics']
print(f"Merged {stats['clusters_merged']} duplicate findings")
print(f"Reduced from {stats['total_findings_before']} to {stats['total_findings_after']}")

# Access deduplicated findings
for finding in result['deduplicated_findings']:
    if finding['is_merged']:
        print(f"Cluster: {finding['personas']}")
        print(f"Category: {finding['category']}, Severity: {finding['severity']}")
```

### Custom Configuration

```python
from referee._utils.deduplicator import get_dedup_config

# Check current config
config = get_dedup_config()
print(config)  # {'enabled': True, 'similarity_threshold': 0.8, ...}

# Override for specific run
result = deduplicate_findings(
    reports=reports,
    paper_text=paper_text,
    similarity_threshold=0.6,  # More aggressive merging
    preserve_distinct=False    # Merge even with severity differences
)
```

## Output

### UI Display

The Streamlit UI shows deduplication statistics in the metadata section:

```
🔗 Cross-Reference Deduplication

Deduplication Statistics:
  Findings Before: 24
  Findings After: 18
  Clusters Merged: 6
  Reduction Rate: 25.0%

Configuration:
  Semantic Embeddings: Available ✓
  Method: Multi-metric (semantic + keywords)

🔍 View Merged Findings (6 clusters)
```

### Excel Export

Two sheets are added to the Excel export:

**"Deduplication" sheet:**
| Metric | Value |
|--------|-------|
| Findings Before Deduplication | 24 |
| Findings After Deduplication | 18 |
| Clusters Merged | 6 |
| Reduction Rate (%) | 25.0% |
| Semantic Embeddings Available | Yes |
| Similarity Method | Multi-metric (semantic + keywords) |

**"Dedup Details" sheet:**
| Personas | Category | Severity | Is Merged | Source Count | Finding Text |
|----------|----------|----------|-----------|--------------|--------------|
| Theorist, Empiricist | identification | fatal | Yes | 2 | **[Identified by: ...]** |
| ... | ... | ... | ... | ... | ... |

### Markdown Report

A "Round 2.5" section is added between Round 2C and Round 3:

```markdown
## ROUND 2.5: CROSS-REFERENCE DEDUPLICATION

**Deduplication Summary:**
- Findings Before: 24
- Findings After: 18
- Clusters Merged: 6
- Reduction Rate: 25.0%
- Method: Multi-metric (semantic + keywords)

**Merged Finding Clusters (6):**

#### Cluster 1
- **Identified by:** Theorist, Empiricist
- **Category:** identification | **Severity:** fatal
- **Similarity Score:** 0.87

[Merged finding text...]
```

## Performance

**Typical Statistics:**
- Reduction rate: 20-40% (depends on panel overlap)
- Processing time: ~2-5 seconds for 3 personas
- No LLM calls (pure algorithmic clustering)
- No additional API costs

**Caching:**
Deduplication results are NOT cached (always recomputed) since they depend on the exact Round 2C reports, which are already cached.

## Best Practices

### When to Lower the Threshold

Lower the similarity threshold (e.g., 0.6-0.7) when:
- Personas use very different terminology for the same issue
- You want more aggressive deduplication
- The paper has many redundant findings

### When to Disable preserve_distinct

Set `preserve_distinct=false` when:
- You want maximum deduplication regardless of severity
- You're okay merging a fatal finding with a moderate one
- The panel tends to over-report severity

### When to Disable Deduplication

Set `ENABLE_DEDUPLICATION=false` when:
- You want to see all raw findings without any merging
- You're debugging the debate process
- You suspect over-merging is losing information

## Transparency and Auditing

All deduplication decisions are logged and tracked:

```python
# Full cluster information is stored
clusters = result['clusters']

for cluster in clusters:
    print(f"Cluster with {len(cluster.findings)} findings:")
    print(f"  Average similarity: {cluster.avg_similarity:.3f}")
    for finding in cluster.findings:
        print(f"    - {finding.persona}: {finding.text[:100]}...")
```

The system prints deduplication progress during execution:

```
[Deduplicator] Starting deduplication (threshold=0.8, preserve_distinct=True)
[Deduplicator] Extracted 8 findings from Empiricist
[Deduplicator] Extracted 9 findings from Theorist
[Deduplicator] Extracted 7 findings from Historian
[Deduplicator] Calculating similarities for 24 findings...
[Deduplicator] Created 18 clusters from 24 findings
[Deduplicator] Merged 6 duplicate findings
[Deduplicator] Complete: 24 findings → 18 findings (6 merged)
```

## Troubleshooting

### No Findings Merged

If `clusters_merged = 0`:

1. **Check similarity threshold:** Try lowering to 0.6-0.7
2. **Check preserve_distinct:** Try setting to `false`
3. **Check finding extraction:** Ensure findings are being extracted correctly
4. **Check persona variety:** Very different personas may not overlap

### Over-Merging

If too many findings are merged incorrectly:

1. **Raise threshold:** Try 0.85 or 0.9
2. **Enable preserve_distinct:** Set to `true`
3. **Install sentence-transformers:** Semantic similarity is more accurate

### Semantic Embeddings Not Loading

If embeddings show as "Not Available":

```bash
# Install sentence-transformers
pip install sentence-transformers

# Test installation
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); print('OK')"
```

On first run, the model will download (~500MB). Subsequent runs are fast.

## Related Documentation

- [Referee System Overview](../README.md)
- [Quote Validation](quote_validation.md) - Prevents hallucinations in quotes
- [Caching System](caching.md) - Speeds up repeated evaluations
- [Changelog](changelog.md) - Version history

## Implementation Details

**Key Functions:**

- `deduplicate_findings()` - Main pipeline
- `cluster_similar_findings()` - Greedy clustering algorithm
- `merge_cluster()` - Consolidates cluster into single finding
- `identify_cross_references()` - Detects explicit cross-references
- `_calculate_similarity()` - Multi-metric similarity scoring

**Data Structures:**

- `Finding` - Represents a single finding with metadata
- `FindingCluster` - Group of similar findings
- Similarity matrix: NxN array of pairwise scores

**Algorithm Complexity:**
- Extraction: O(n) where n = report length
- Similarity: O(k²) where k = number of findings (~20-30)
- Clustering: O(k² log k) for sorting pairs
- Overall: O(k²) - very fast for typical k < 50

## Testing

Run the test suite:

```bash
cd app_system
python tests/test_deduplicator.py
```

Tests cover:
- Finding extraction from reports
- Similarity calculation
- Clustering algorithm
- Merge strategy
- Full pipeline integration
- Configuration loading

## Future Enhancements

Potential improvements (not yet implemented):

1. **Vision-based similarity:** Compare figures cited in findings
2. **Context-aware merging:** Use paper structure to inform merging
3. **User feedback:** Learn from user corrections to merging decisions
4. **Cross-round deduplication:** Also deduplicate Round 1 findings
5. **Explainable similarity:** Show why findings were merged (feature attribution)

## Contact

For questions or issues with deduplication:
- File an issue on GitHub
- Check the test suite for usage examples
- Review the deduplicator.py source code (well-documented)
