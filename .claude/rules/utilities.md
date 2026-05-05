# Utilities & Infrastructure Context

This rule activates when working on shared utilities, caching, deduplication, or infrastructure.

## File Scope
- `app_system/utils.py` — Core LLM utilities
- `app_system/config.py` — API configuration
- `app_system/referee/_utils/` — Referee-specific utilities
- `app_system/section_eval/utils.py` — Section eval utilities

## Core Utilities (`app_system/utils.py`)

### LLM Call Patterns

**Stateless (Referee System)**:
```python
def single_query(prompt: str, temperature: float = 1.0) -> str:
    """
    Stateless LLM call with thinking mode enabled.
    - Temperature: 1.0 (required for thinking)
    - Thinking budget: 2048 tokens
    - Retries: 3× with 5s delay
    - Model: MODEL_PRIMARY (Claude 4.5 Sonnet)
    """
```

**Stateful (Section Evaluator)**:
```python
class ConversationManager:
    def conv_query(self, prompt: str) -> str:
        """
        Stateful conversation with auto-pruning.
        - Auto-prunes at 8000 tokens
        - Summarizes old messages
        - Preserves context across queries
        """
```

**Direct API Call (Section Evaluator)**:
```python
def safe_query(prompt: str) -> str:
    """
    Bypass ConversationManager, direct API call.
    - Temperature: 0.3
    - No thinking mode
    - No conversation history
    - Used by section_eval/evaluator.py
    """
```

### Token Counting
```python
def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens using tiktoken."""
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))
```

### Cost Estimation
```python
def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate API cost based on token counts."""
    # Rates vary by model (see config.py)
    return (input_tokens * INPUT_RATE + output_tokens * OUTPUT_RATE) / 1000
```

## API Configuration (`app_system/config.py`)

### Environment Loading
```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env file

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
MODEL_PRIMARY = os.getenv("MODEL_PRIMARY", "anthropic.claude-sonnet-4-5-...")
```

### Model Selection
**CRITICAL**: ALL systems use Claude 4.5 Sonnet.

```python
# Primary model (all main calls)
MODEL_PRIMARY = "anthropic.claude-sonnet-4-5-20250929-v1:0"

# Legacy aliases (deprecated, point to PRIMARY)
MODEL_SECONDARY = MODEL_PRIMARY
MODEL_TERTIARY = MODEL_PRIMARY
```

### Provider Support
```python
# Auto-detect provider from API_BASE
if "anthropic.com" in API_BASE:
    PROVIDER = "anthropic"
elif "openai.com" in API_BASE:
    PROVIDER = "openai"
else:
    PROVIDER = "custom"  # Federal Reserve MartinAI
```

## Caching System (`referee/_utils/cache.py`)

### Architecture
```
.referee_cache/
├── {cache_key}/              # SHA256 hash
│   ├── metadata.json         # Paper info, timestamp
│   ├── round_0_selection.json
│   ├── round_1_reports.json
│   ├── round_2a_cross_exam.json
│   ├── round_2b_answers.json
│   ├── round_2c_amendments.json
│   └── round_3_synthesis.json
```

### Cache Key Computation
```python
def compute_cache_key(
    paper_text: str,
    selected_personas: List[str],
    persona_weights: Dict[str, float],
    model: str,
    paper_type: str = ""
) -> str:
    """
    SHA256 hash of:
    - First 100K chars of paper text
    - Sorted persona list
    - Persona weights
    - Model identifier
    - Paper type
    """
```

### Usage Pattern
```python
from referee._utils.cache import (
    compute_cache_key,
    load_round_results,
    save_round_results,
    check_cache_status
)

# Check if cached
cache_key = compute_cache_key(paper_text, personas, weights, model)
cached_data = load_round_results(cache_key, round_num)

if cached_data:
    return cached_data  # Cache hit
else:
    results = run_round(...)  # Cache miss
    save_round_results(cache_key, round_num, results)
    return results
```

### Cache Management
```python
# Clear cache for specific paper
clear_cache_for_paper(cache_key)

# Clear all cache
clear_cache()

# Get cache stats
stats = get_cache_stats()  # {total_size, num_papers, oldest, newest}
```

### Configuration (.env)
```bash
CACHE_ENABLED=true           # Enable/disable caching
CACHE_TTL_DAYS=30           # Auto-delete after N days
CACHE_DIR=.referee_cache    # Cache directory path
```

## Deduplication System (`referee/_utils/deduplicator.py`)

### Purpose
Identify and merge duplicate findings across persona reports to reduce redundancy.

### Similarity Metrics

**Quote Overlap** (always enabled):
```python
# Findings citing same paper text
similarity = len(quote1 ∩ quote2) / len(quote1 ∪ quote2)
```

**Semantic Similarity** (optional, requires sentence-transformers):
```python
# Embedding-based clustering
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(findings)
similarity = cosine_similarity(embeddings)
```

**Keyword Matching** (always enabled):
```python
# Common technical terms
keywords = extract_keywords(finding)  # Noun phrases, proper nouns
similarity = jaccard(keywords1, keywords2)
```

### Usage
```python
from referee._utils.deduplicator import deduplicate_findings

deduplicated = deduplicate_findings(
    reports={'Empiricist': report1, 'Theorist': report2},
    paper_text=paper_text,
    threshold=0.8,  # Similarity threshold
    preserve_distinct=True  # Keep unique perspectives
)
```

### Configuration (.env)
```bash
ENABLE_DEDUPLICATION=true              # Enable/disable
DEDUP_SIMILARITY_THRESHOLD=0.8         # Merge threshold (0-1)
DEDUP_PRESERVE_DISTINCT_PERSPECTIVES=true  # Keep unique angles
```

### Output
```python
{
    'clusters': [
        {
            'representative': 'Finding from Empiricist...',
            'mentions': ['Empiricist', 'Theorist'],
            'similarity': 0.92
        }
    ],
    'unique_findings': [...]  # Not clustered
}
```

## Quote Validation (`referee/_utils/quote_validator.py`)

### Purpose
Prevent hallucinations by verifying LLM quotes exist in source paper.

### Validation Process

**1. Extract quotes from report**:
```python
# Patterns: "...", '...', > blockquote, "The paper states..."
quotes = extract_quotes(report)
```

**2. Fuzzy match against paper text**:
```python
from thefuzz import fuzz

# Adaptive thresholds
threshold = 95 if is_math_content(quote) else 85
match_score = fuzz.partial_ratio(quote, paper_text)
verified = match_score >= threshold
```

**3. Generate validation report**:
```python
{
    'quote': 'quoted text',
    'verified': True/False,
    'match_score': 87,
    'best_match': 'closest text in paper',
    'location': 'page 5, paragraph 3'
}
```

### Usage
```python
from referee._utils.quote_validator import validate_quotes_in_reports

validation_results = validate_quotes_in_reports(
    reports={'Empiricist': report1, ...},
    paper_text=paper_text
)

# Check results
for persona, results in validation_results.items():
    unverified = [r for r in results if not r['verified']]
    if unverified:
        print(f"{persona} has {len(unverified)} unverified quotes")
```

### Configuration (.env)
```bash
DISABLE_QUOTE_VALIDATION=false  # Set true to disable
```

### Dependencies
```bash
pip install thefuzz python-Levenshtein  # Levenshtein optional but faster
```

## PDF Extraction (`referee/_utils/pdf_extractor_v2.py`)

### Features
- **PyMuPDF-based** (fitz library)
- **Multi-column layout handling**
- **Figure extraction** (embedded images + rendered vector graphics)
- **Table extraction** with OCR
- **Caption parsing** with multi-panel detection
- **Fallback to pdfplumber** if PyMuPDF unavailable

### Usage
```python
from referee._utils.pdf_extractor_v2 import extract_pdf_with_figures, PYMUPDF_AVAILABLE

if PYMUPDF_AVAILABLE:
    result = extract_pdf_with_figures(pdf_bytes)
    text = result.text
    figures = result.figures  # List[Figure]
    tables = result.tables    # List[Table]
else:
    # Fallback to pdfplumber
    text = extract_with_pdfplumber(pdf_bytes)
```

### Figure Object
```python
@dataclass
class Figure:
    figure_number: str  # "1", "3a"
    figure_id: str      # "figure_1", "figure_3a"
    page_number: int    # 1-indexed
    image_data: bytes   # PNG/JPEG
    caption: str        # Full caption text
    panels: List[str]   # ["a", "b"] if multi-panel
```

### Dependencies
```bash
pip install pymupdf        # Required
pip install Pillow         # Required for images
pip install pytesseract    # Optional for OCR
```

## Common Patterns

### Error Handling
```python
try:
    result = single_query(prompt)
except Exception as e:
    logger.error(f"LLM call failed: {e}")
    # Retry logic (3× with backoff)
    for attempt in range(3):
        time.sleep(5 * (attempt + 1))
        try:
            result = single_query(prompt)
            break
        except:
            if attempt == 2:
                raise
```

### Thread Safety
```python
from threading import Lock

cache_lock = Lock()

def save_to_cache(key, value):
    with cache_lock:
        # Atomic write operation
        cache[key] = value
```

### Async Operations
```python
async def process_parallel(items):
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results
```

## Testing Utilities

```bash
cd app_system
python -m pytest tests/test_cache.py
python -m pytest tests/test_deduplicator.py
python -m pytest tests/test_quote_validator.py
python -m pytest tests/test_pymupdf_extractor.py
```

## Performance Optimization

### Caching Strategy
- **Cache hits**: ~50-80% savings during development
- **Granularity**: Per-round (not full debate)
- **Invalidation**: SHA256 hash changes → cache miss

### Parallel Execution
```python
# Run personas in parallel (4× speedup for 4 personas)
tasks = [call_llm_async(persona, prompt) for persona in personas]
results = await asyncio.gather(*tasks)
```

### Token Optimization
- **Summarization**: Compress long reports before injecting into later rounds
- **Pruning**: Remove low-value content (boilerplate, excessive examples)
- **Truncation**: Limit paper text to first 50K tokens for cache key

## Common Pitfalls

❌ **Don't** bypass `single_query()` for referee calls (breaks thinking mode)
❌ **Don't** mix `safe_query()` and `ConversationManager` in same workflow
❌ **Don't** disable quote validation without user consent
❌ **Don't** forget to increment cache prefix after schema changes
✅ **Do** use async for parallel LLM calls
✅ **Do** log cache hit rates for optimization
✅ **Do** validate configuration on startup
✅ **Do** handle API rate limits gracefully
