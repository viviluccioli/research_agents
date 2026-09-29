"""
Cross-Reference Deduplication Module for Referee Reports

Identifies and merges duplicate findings across persona reports to reduce redundancy
while preserving unique perspectives. Uses multiple similarity metrics including:
- Quote overlap: Findings citing the same paper section
- Semantic similarity: Embedding-based clustering (optional, requires sentence-transformers)
- Category/severity overlap: Same issue type and severity
- Keyword matching: Common technical terms and phrases

Configuration:
- Set ENABLE_DEDUPLICATION=true/false in .env (default: true)
- Set DEDUP_SIMILARITY_THRESHOLD in .env (default: 0.8)
- Set DEDUP_PRESERVE_DISTINCT_PERSPECTIVES=true/false in .env (default: true)

Usage:
    from referee._utils.deduplicator import deduplicate_findings

    deduplicated = deduplicate_findings(
        reports={'Empiricist': report_text, ...},
        paper_text=paper_text
    )
"""

import re
import os
import logging
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field

# Try to import sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.info("sentence-transformers not available. Using keyword-based similarity only.")

# Configuration from environment
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_PRESERVE_DISTINCT = True

# Severity levels for categorization
SEVERITY_KEYWORDS = {
    'fatal': ['fatal', 'critical', 'severe', 'major', 'fundamental', 'invalidate', 'broken'],
    'major': ['significant', 'important', 'substantial', 'concerning', 'problematic'],
    'moderate': ['moderate', 'noticeable', 'minor issue', 'improvement needed'],
    'minor': ['minor', 'small', 'trivial', 'cosmetic', 'suggestion']
}

# Category keywords for grouping
CATEGORY_KEYWORDS = {
    'identification': ['identification', 'endogeneity', 'causality', 'causal', 'instrument', 'IV', 'exogeneity'],
    'data': ['data', 'dataset', 'sample', 'measurement', 'variable', 'observation'],
    'methodology': ['method', 'econometric', 'specification', 'model', 'estimation', 'regression'],
    'theory': ['theory', 'theoretical', 'model', 'assumption', 'proposition', 'proof'],
    'literature': ['literature', 'citation', 'reference', 'prior work', 'existing research'],
    'interpretation': ['interpretation', 'conclusion', 'implication', 'finding', 'result'],
    'presentation': ['clarity', 'writing', 'presentation', 'organization', 'explanation'],
    'robustness': ['robustness', 'sensitivity', 'alternative', 'specification check'],
}


@dataclass
class Finding:
    """Represents a single finding/issue identified in a report."""
    text: str
    persona: str
    severity: str = 'unknown'
    category: str = 'general'
    quotes: List[str] = field(default_factory=list)
    keywords: Set[str] = field(default_factory=set)
    embedding: Optional[any] = None

    def __post_init__(self):
        """Extract metadata after initialization."""
        if not self.severity or self.severity == 'unknown':
            self.severity = self._extract_severity()
        if not self.category or self.category == 'general':
            self.category = self._extract_category()
        if not self.quotes:
            self.quotes = self._extract_quotes()
        if not self.keywords:
            self.keywords = self._extract_keywords()

    def _extract_severity(self) -> str:
        """Extract severity from finding text."""
        text_lower = self.text.lower()

        # Check for explicit severity markers
        if re.search(r'\[fatal\]|\[critical\]|⚠️|❌', text_lower):
            return 'fatal'

        # Check severity keywords
        for severity, keywords in SEVERITY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return severity

        return 'moderate'  # Default

    def _extract_category(self) -> str:
        """Extract category from finding text."""
        text_lower = self.text.lower()

        # Count matches per category
        category_scores = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                category_scores[category] = score

        # Return category with highest score
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]

        return 'general'

    def _extract_quotes(self) -> List[str]:
        """Extract quoted text from finding."""
        quotes = []

        # Pattern 1: Text in quotes
        quote_pattern = r'"([^"]{10,200})"'
        quotes.extend(re.findall(quote_pattern, self.text))

        # Pattern 2: Text in blockquotes
        blockquote_pattern = r'^>\s*(.+)$'
        quotes.extend(re.findall(blockquote_pattern, self.text, re.MULTILINE))

        return quotes

    def _extract_keywords(self) -> Set[str]:
        """Extract technical keywords from finding."""
        # Remove quoted text to avoid double-counting
        text = re.sub(r'"[^"]+"', '', self.text)

        # Common technical terms in economics papers
        keywords = set()

        # Statistical terms
        stats_pattern = r'\b(regression|coefficient|standard error|p-value|significance|confidence interval|hypothesis)\b'
        keywords.update(re.findall(stats_pattern, text, re.IGNORECASE))

        # Econometric terms
        econ_pattern = r'\b(endogeneity|instrument|panel|fixed effects?|random effects?|heteroskedasticity|autocorrelation)\b'
        keywords.update(re.findall(econ_pattern, text, re.IGNORECASE))

        # Research terms
        research_pattern = r'\b(identification|causality|mechanism|robustness|specification|estimation)\b'
        keywords.update(re.findall(research_pattern, text, re.IGNORECASE))

        return {kw.lower() for kw in keywords}


@dataclass
class FindingCluster:
    """Represents a cluster of similar findings."""
    findings: List[Finding]
    representative: Optional[Finding] = None
    merged_text: str = ""
    personas: List[str] = field(default_factory=list)
    avg_similarity: float = 0.0

    def __post_init__(self):
        """Initialize cluster properties."""
        self.personas = [f.persona for f in self.findings]
        if self.findings and not self.representative:
            # Use finding with highest severity as representative
            self.representative = max(self.findings, key=lambda f: self._severity_rank(f.severity))

    @staticmethod
    def _severity_rank(severity: str) -> int:
        """Convert severity to numeric rank."""
        ranks = {'fatal': 4, 'major': 3, 'moderate': 2, 'minor': 1, 'unknown': 0}
        return ranks.get(severity, 0)


def get_dedup_config() -> Dict[str, any]:
    """Load deduplication configuration from environment."""
    return {
        'enabled': os.environ.get('ENABLE_DEDUPLICATION', 'true').lower() in ('true', '1', 'yes'),
        'similarity_threshold': float(os.environ.get('DEDUP_SIMILARITY_THRESHOLD', str(DEFAULT_SIMILARITY_THRESHOLD))),
        'preserve_distinct': os.environ.get('DEDUP_PRESERVE_DISTINCT_PERSPECTIVES', 'true').lower() in ('true', '1', 'yes'),
    }


def _extract_findings_from_report(report_text: str, persona: str) -> List[Finding]:
    """
    Extract individual findings from a persona's report.

    Args:
        report_text: The full report text
        persona: Name of the persona

    Returns:
        List of Finding objects
    """
    findings = []

    # Strategy 1: Look for bullet points with severity markers
    bullet_pattern = r'^[\s]*[-•*]\s*(.+?)(?=\n[-•*]|\n\n|\Z)'
    bullets = re.findall(bullet_pattern, report_text, re.MULTILINE | re.DOTALL)

    for bullet in bullets:
        bullet = bullet.strip()
        # Skip very short bullets (likely not findings)
        if len(bullet) < 30:
            continue
        # Skip if it's just a heading or meta-text
        if re.match(r'^\*\*[A-Z]', bullet) or 'verdict' in bullet.lower()[:20]:
            continue

        finding = Finding(text=bullet, persona=persona)
        findings.append(finding)

    # Strategy 2: Look for numbered issues
    numbered_pattern = r'(?:^|\n)\s*\d+\.\s*(.+?)(?=\n\d+\.|\n\n|\Z)'
    numbered = re.findall(numbered_pattern, report_text, re.DOTALL)

    for item in numbered:
        item = item.strip()
        if len(item) < 30:
            continue
        # Avoid duplicates from bullets
        if not any(item[:50] in f.text for f in findings):
            finding = Finding(text=item, persona=persona)
            findings.append(finding)

    # Strategy 3: Look for paragraph-level issues (fallback)
    if not findings:
        # Split into paragraphs
        paragraphs = [p.strip() for p in report_text.split('\n\n') if len(p.strip()) > 50]

        # Filter to paragraphs that likely contain findings
        for para in paragraphs:
            # Skip meta-sections
            if any(x in para[:50].lower() for x in ['verdict', 'summary', 'overall', 'recommendation']):
                continue

            # Look for issue indicators
            if any(indicator in para.lower() for indicator in ['issue', 'problem', 'concern', 'weakness', 'flaw', 'error', 'incorrect', 'missing', 'lack']):
                finding = Finding(text=para, persona=persona)
                findings.append(finding)

    return findings


def _calculate_quote_overlap(f1: Finding, f2: Finding, paper_text: str) -> float:
    """
    Calculate overlap based on quoted text from paper.

    Returns:
        Overlap score between 0.0 and 1.0
    """
    if not f1.quotes or not f2.quotes:
        return 0.0

    # Normalize quotes
    quotes1 = {_normalize_text(q) for q in f1.quotes}
    quotes2 = {_normalize_text(q) for q in f2.quotes}

    # Calculate Jaccard similarity
    if not quotes1 or not quotes2:
        return 0.0

    intersection = len(quotes1 & quotes2)
    union = len(quotes1 | quotes2)

    return intersection / union if union > 0 else 0.0


def _calculate_keyword_overlap(f1: Finding, f2: Finding) -> float:
    """
    Calculate overlap based on technical keywords.

    Returns:
        Overlap score between 0.0 and 1.0
    """
    if not f1.keywords or not f2.keywords:
        return 0.0

    intersection = len(f1.keywords & f2.keywords)
    union = len(f1.keywords | f2.keywords)

    return intersection / union if union > 0 else 0.0


def _calculate_category_similarity(f1: Finding, f2: Finding) -> float:
    """
    Calculate similarity based on category and severity.

    Returns:
        Score between 0.0 and 1.0
    """
    category_match = 1.0 if f1.category == f2.category else 0.0
    severity_match = 1.0 if f1.severity == f2.severity else 0.5  # Partial credit for different severity

    # Weight: 70% category, 30% severity
    return 0.7 * category_match + 0.3 * severity_match


def _calculate_semantic_similarity(f1: Finding, f2: Finding, model: any) -> float:
    """
    Calculate semantic similarity using embeddings.

    Args:
        f1, f2: Findings to compare
        model: SentenceTransformer model

    Returns:
        Cosine similarity between 0.0 and 1.0
    """
    if not EMBEDDINGS_AVAILABLE or model is None:
        return 0.0

    if f1.embedding is None:
        f1.embedding = model.encode(f1.text, convert_to_tensor=False)
    if f2.embedding is None:
        f2.embedding = model.encode(f2.text, convert_to_tensor=False)

    # Cosine similarity
    import numpy as np
    dot_product = np.dot(f1.embedding, f2.embedding)
    norm1 = np.linalg.norm(f1.embedding)
    norm2 = np.linalg.norm(f2.embedding)

    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    normalized = re.sub(r'\s+', ' ', text.strip())
    return normalized.lower()


def _calculate_similarity(f1: Finding, f2: Finding, paper_text: str, model: any = None) -> float:
    """
    Calculate overall similarity between two findings using multiple metrics.

    Args:
        f1, f2: Findings to compare
        paper_text: Full paper text (for quote validation)
        model: Optional SentenceTransformer model for semantic similarity

    Returns:
        Combined similarity score between 0.0 and 1.0
    """
    # Don't merge findings from the same persona
    if f1.persona == f2.persona:
        return 0.0

    # Calculate individual metrics
    quote_overlap = _calculate_quote_overlap(f1, f2, paper_text)
    keyword_overlap = _calculate_keyword_overlap(f1, f2)
    category_sim = _calculate_category_similarity(f1, f2)

    # Semantic similarity (if available)
    semantic_sim = 0.0
    if EMBEDDINGS_AVAILABLE and model is not None:
        semantic_sim = _calculate_semantic_similarity(f1, f2, model)

    # Weighted combination
    if EMBEDDINGS_AVAILABLE and model is not None:
        # With embeddings: 40% semantic, 25% keywords, 20% quotes, 15% category
        weights = [0.40, 0.25, 0.20, 0.15]
        scores = [semantic_sim, keyword_overlap, quote_overlap, category_sim]
    else:
        # Without embeddings: 45% keywords, 30% category, 25% quotes
        weights = [0.45, 0.30, 0.25]
        scores = [keyword_overlap, category_sim, quote_overlap]

    combined = sum(w * s for w, s in zip(weights, scores))
    return combined


def cluster_similar_findings(
    findings: List[Finding],
    paper_text: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    preserve_distinct: bool = DEFAULT_PRESERVE_DISTINCT
) -> List[FindingCluster]:
    """
    Cluster similar findings using similarity metrics.

    Args:
        findings: List of findings to cluster
        paper_text: Full paper text
        similarity_threshold: Minimum similarity to merge (0.0-1.0)
        preserve_distinct: If True, preserve findings with different severity/perspective

    Returns:
        List of FindingCluster objects
    """
    if not findings:
        return []

    # Load embedding model if available
    model = None
    if EMBEDDINGS_AVAILABLE:
        try:
            print("[Deduplicator] Loading sentence-transformers model...")
            model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model
            print("[Deduplicator] Model loaded successfully")
        except Exception as e:
            print(f"[Deduplicator] Failed to load embedding model: {e}")
            model = None

    # Calculate pairwise similarities
    n = len(findings)
    similarity_matrix = [[0.0] * n for _ in range(n)]

    print(f"[Deduplicator] Calculating similarities for {n} findings...")
    for i in range(n):
        for j in range(i + 1, n):
            sim = _calculate_similarity(findings[i], findings[j], paper_text, model)
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim

    # Greedy clustering: start with highest similarities
    clusters = []
    assigned = set()

    # Get all pairs with similarity above threshold
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] >= similarity_threshold:
                pairs.append((i, j, similarity_matrix[i][j]))

    # Sort by similarity (highest first)
    pairs.sort(key=lambda x: x[2], reverse=True)

    # Build clusters
    cluster_map = {}  # finding_idx -> cluster_idx

    for i, j, sim in pairs:
        # Check if either finding is already assigned
        cluster_i = cluster_map.get(i)
        cluster_j = cluster_map.get(j)

        if preserve_distinct:
            # Don't merge if severity difference is too large
            severity_diff = abs(
                FindingCluster._severity_rank(findings[i].severity) -
                FindingCluster._severity_rank(findings[j].severity)
            )
            if severity_diff >= 2:  # e.g., fatal vs moderate
                continue

        if cluster_i is None and cluster_j is None:
            # Create new cluster
            new_cluster = FindingCluster(findings=[findings[i], findings[j]])
            clusters.append(new_cluster)
            cluster_map[i] = len(clusters) - 1
            cluster_map[j] = len(clusters) - 1
        elif cluster_i is not None and cluster_j is None:
            # Add j to i's cluster
            clusters[cluster_i].findings.append(findings[j])
            cluster_map[j] = cluster_i
        elif cluster_i is None and cluster_j is not None:
            # Add i to j's cluster
            clusters[cluster_j].findings.append(findings[i])
            cluster_map[i] = cluster_j
        elif cluster_i != cluster_j:
            # Merge two clusters
            larger = cluster_i if len(clusters[cluster_i].findings) >= len(clusters[cluster_j].findings) else cluster_j
            smaller = cluster_j if larger == cluster_i else cluster_i

            # Add all findings from smaller to larger
            clusters[larger].findings.extend(clusters[smaller].findings)

            # Update cluster map
            for idx, cluster_idx in list(cluster_map.items()):
                if cluster_idx == smaller:
                    cluster_map[idx] = larger

    # Add unclustered findings as singleton clusters
    for i in range(n):
        if i not in cluster_map:
            clusters.append(FindingCluster(findings=[findings[i]]))

    # Calculate average similarity for each cluster
    for cluster in clusters:
        if len(cluster.findings) > 1:
            sims = []
            for i, f1 in enumerate(cluster.findings):
                for f2 in cluster.findings[i+1:]:
                    idx1 = findings.index(f1)
                    idx2 = findings.index(f2)
                    sims.append(similarity_matrix[idx1][idx2])
            cluster.avg_similarity = sum(sims) / len(sims) if sims else 0.0

    print(f"[Deduplicator] Created {len(clusters)} clusters from {n} findings")
    print(f"[Deduplicator] Merged {n - len(clusters)} duplicate findings")

    return clusters


def merge_cluster(cluster: FindingCluster) -> Dict[str, any]:
    """
    Merge a cluster of findings into a single consolidated finding.

    Args:
        cluster: FindingCluster to merge

    Returns:
        Dictionary with merged finding information
    """
    if len(cluster.findings) == 1:
        # No merging needed
        f = cluster.findings[0]
        return {
            'text': f.text,
            'personas': [f.persona],
            'severity': f.severity,
            'category': f.category,
            'is_merged': False,
            'source_count': 1,
            'quotes': f.quotes,
        }

    # Multiple findings - merge them
    rep = cluster.representative

    # Collect unique perspectives
    perspectives = []
    for f in cluster.findings:
        # Extract key sentence (first sentence usually)
        sentences = re.split(r'[.!?]\s+', f.text)
        if sentences:
            perspectives.append(f"{f.persona}: {sentences[0]}")

    # Build merged text
    merged_text = f"**[Identified by: {', '.join(cluster.personas)}]**\n\n"
    merged_text += f"{rep.text}\n\n"

    if len(perspectives) > 1:
        merged_text += "**Additional Perspectives:**\n"
        for p in perspectives[1:]:  # Skip representative (already included)
            merged_text += f"- {p}\n"

    # Collect all unique quotes
    all_quotes = []
    seen_quotes = set()
    for f in cluster.findings:
        for q in f.quotes:
            norm = _normalize_text(q)
            if norm not in seen_quotes:
                seen_quotes.add(norm)
                all_quotes.append(q)

    return {
        'text': merged_text,
        'personas': cluster.personas,
        'severity': rep.severity,
        'category': rep.category,
        'is_merged': True,
        'source_count': len(cluster.findings),
        'quotes': all_quotes,
        'similarity_score': cluster.avg_similarity,
    }


def deduplicate_findings(
    reports: Dict[str, str],
    paper_text: str,
    similarity_threshold: Optional[float] = None,
    preserve_distinct: Optional[bool] = None
) -> Dict[str, any]:
    """
    Main deduplication pipeline: extract, cluster, and merge findings.

    Args:
        reports: Dictionary mapping persona names to their Round 2C reports
        paper_text: Full paper text
        similarity_threshold: Override default threshold
        preserve_distinct: Override default preserve_distinct setting

    Returns:
        Dictionary with:
        - deduplicated_findings: List of merged findings
        - statistics: Deduplication statistics
        - clusters: Raw cluster information (for debugging)
    """
    # Load config
    config = get_dedup_config()

    if not config['enabled']:
        print("[Deduplicator] Deduplication disabled by configuration")
        return {
            'deduplicated_findings': [],
            'statistics': {
                'enabled': False,
                'total_findings_before': 0,
                'total_findings_after': 0,
                'clusters_merged': 0,
            },
            'clusters': []
        }

    # Use provided params or config defaults
    threshold = similarity_threshold if similarity_threshold is not None else config['similarity_threshold']
    preserve = preserve_distinct if preserve_distinct is not None else config['preserve_distinct']

    print(f"[Deduplicator] Starting deduplication (threshold={threshold}, preserve_distinct={preserve})")

    # Extract findings from all reports
    all_findings = []
    for persona, report in reports.items():
        findings = _extract_findings_from_report(report, persona)
        print(f"[Deduplicator] Extracted {len(findings)} findings from {persona}")
        all_findings.extend(findings)

    total_before = len(all_findings)

    if total_before == 0:
        print("[Deduplicator] No findings extracted, skipping deduplication")
        return {
            'deduplicated_findings': [],
            'statistics': {
                'enabled': True,
                'total_findings_before': 0,
                'total_findings_after': 0,
                'clusters_merged': 0,
                'embeddings_available': EMBEDDINGS_AVAILABLE,
            },
            'clusters': []
        }

    # Cluster similar findings
    clusters = cluster_similar_findings(
        all_findings,
        paper_text,
        similarity_threshold=threshold,
        preserve_distinct=preserve
    )

    # Merge clusters
    merged_findings = []
    for cluster in clusters:
        merged = merge_cluster(cluster)
        merged_findings.append(merged)

    total_after = len(merged_findings)
    clusters_merged = total_before - total_after

    # Calculate statistics by category
    category_stats = defaultdict(lambda: {'before': 0, 'after': 0})
    for f in all_findings:
        category_stats[f.category]['before'] += 1
    for mf in merged_findings:
        category_stats[mf['category']]['after'] += 1

    statistics = {
        'enabled': True,
        'total_findings_before': total_before,
        'total_findings_after': total_after,
        'clusters_merged': clusters_merged,
        'reduction_rate': round((clusters_merged / total_before * 100), 1) if total_before > 0 else 0.0,
        'embeddings_available': EMBEDDINGS_AVAILABLE,
        'similarity_threshold': threshold,
        'preserve_distinct_perspectives': preserve,
        'category_breakdown': dict(category_stats),
    }

    print(f"[Deduplicator] Complete: {total_before} findings → {total_after} findings ({clusters_merged} merged)")

    return {
        'deduplicated_findings': merged_findings,
        'statistics': statistics,
        'clusters': clusters,  # Include for debugging/transparency
    }


def identify_cross_references(
    findings: List[Dict],
    paper_text: str
) -> Dict[str, List[str]]:
    """
    Identify explicit cross-references between findings.

    This looks for cases where one persona explicitly references
    another's finding (e.g., "As the Empiricist noted...").

    Args:
        findings: List of merged findings
        paper_text: Full paper text

    Returns:
        Dictionary mapping finding indices to lists of related finding indices
    """
    cross_refs = defaultdict(list)

    # Persona names to look for
    persona_names = ['Theorist', 'Empiricist', 'Historian', 'Visionary', 'Policymaker']

    for i, finding in enumerate(findings):
        text_lower = finding['text'].lower()

        # Look for explicit references to other personas
        for persona in persona_names:
            if persona.lower() in text_lower:
                # This finding references another persona
                # Try to find which finding it's referencing
                for j, other_finding in enumerate(findings):
                    if i != j and persona in other_finding['personas']:
                        cross_refs[i].append(j)

        # Look for phrases like "similarly", "also", "additionally"
        if any(phrase in text_lower for phrase in ['similarly', 'also problematic', 'additionally', 'related concern']):
            # This might be a cross-reference
            # Mark for manual review
            pass

    return dict(cross_refs)
