# exp-12: specialist peer review

The engine selects three complementary specialists, collects independent PRE
assessments, runs synchronous issue-centered debate, collects independent POST
assessments, and asks an editor for a qualitative verdict and author letter.
It saves raw judgments and their history. It computes no paper-quality score,
persona weights, credibility adjustments, or publication-tier calibration.

## Files

- `exp-12.py`: roles, prompts, examples, versioned ledger, model calls, workflow,
  report generation and CLI.
- `config.py`: the single default configuration and optional override loader.
- `schemas.py`: validated output and saved-record contracts; JSON Schema generation.
- `token_tracker.py`: per-attempt usage and optional cost accounting.
- `retrieval.py`: mechanical manuscript chunks, deterministic queries and evidence selection.
- `pyproject.toml`: installation and dependency declarations.
- `tests/`: offline tests covering contracts, retrieval, ledger and full workflow.
- `exp_12_instructions.md`: original research/architecture specification.
- `exp_12_proposal.md`: now the changelog, including later decisions that supersede
  the original proposal. Its filename remains stable for existing links.

There are no nested helper packages, duplicated YAML settings or import bridges.

## Install and verify

Use Python 3.9 or newer. Download this folder together; install from it:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m unittest discover -s tests -v
python exp-12.py --help
```

`pip install -r requirements.txt` is an equivalent installation shortcut.
The package installs the four helper modules and the `exp-12.py` command.
Tests use a fake client and temporary output directories. They need no API key
and make no network calls. Importing the modules creates no output directories.

## Run a manuscript

Input is UTF-8 text, not a PDF. Configure a chat-completions-compatible endpoint
and a model it actually supports:

```sh
export PEER_REVIEW_API_BASE="https://your-provider.example/v1"
export PEER_REVIEW_MODEL="your-model-id"
# Set PEER_REVIEW_API_KEY through your shell or secret manager if the endpoint requires it.
python exp-12.py /path/to/manuscript.txt --output-dir /path/to/reviews
```

`OPENAI_BASE_URL` and `OPENAI_API_KEY` are accepted environment fallbacks.
Local endpoints may operate without a key. Editing, imports, and offline tests
never need credentials. No credentials are read from IDE authentication files,
stored in config, or included in saved requests.

The inherited default model identifier in `config.py` is not a claim of provider
availability. Choose the endpoint/model explicitly before live acceptance.
The transport sends `reasoning_effort: "none"`; it does not silently switch to a
different reasoning mode or temperature if a provider rejects that setting.
Provider schema support is not assumed: model outputs are validated locally.

Defaults: two debate rounds; temperature 0 for selection/PRE/POST/editor, 0.35
for debate; reasoning off. A complete two-round run uses 14 successful calls
plus any retries. `--rounds 0` permits the independent-panel comparison with POST
and editor but without debate (8 successful calls).

`--model` overrides the default model; explicit per-stage models take precedence.
`PEER_REVIEW_MODEL` also overrides the configured default. For an optional custom
configuration, pass `--config /path/to/settings.json`; only the supplied fields
override defaults. Example content:

```json
{
  "default_model": "your-model-id",
  "debate_rounds": 2,
  "post_manuscript_context": "retrieved",
  "retrieval_method": "lexical",
  "retrieval_budget_characters": 12000,
  "stages": {"debate": {"temperature": 0.35}},
  "max_attempts": 3
}
```

An invalid or missing explicit settings file fails clearly. There are no
persona-specific settings. Pricing, if needed, is an optional model-keyed
`pricing` object in settings with `input`, `output`, optional `cache_read` and
`cache_write` rates per million tokens, plus a required descriptive `basis`.

## POST manuscript evidence

`post_manuscript_context` supports `retrieved` (default), `abstract_only` (the prior
context shape), and `full_manuscript` (an ablation, still bounded by
`manuscript_characters`). The abstract and complete frozen debate record remain available.

Retrieval mechanically splits the original text at paragraph, sentence or word
boundaries, without identifying semantic sections. Chunk IDs, exact character offsets,
lengths and neighbors are preserved. Queries come from current active concerns,
current-version challenges/defenses, and the reviewer's own PRE concerns and judgments.
No extra review-model call generates queries or summarizes manuscript passages.

The runnable default is lexical BM25. `embedding` and `hybrid` are available through
an explicitly supplied `embedding_backend` argument to `run_review`. A backend must
expose a stable nonempty `fingerprint` identifying its model/revision/settings and an
`embed(texts)` method returning one finite, nonzero numeric vector per input. Use a
deterministic backend. Exp-12 does not install or download an embedding model, reuse
chat credentials for embeddings, or silently substitute lexical retrieval when a
semantic backend is unavailable. Choosing an embedding model remains a separate decision.
Runtime dependencies remain only pydantic and requests.

BM25 and cosine rankings use deterministic reciprocal-rank fusion and stable chunk-ID
tie breaking; selection suppresses near-duplicate word sets. Chunks have no overlap.
The default chunk target is 1,600 characters (smaller for short manuscripts).
`retrieval_budget_characters` defaults to 12,000 characters for the serialized evidence
array, including provenance. Raw passage text also cannot exceed
`retrieval_max_manuscript_fraction` of the manuscript (default 0.25). Selection keeps
whole chunks; a chunk that cannot fit is skipped. Small budgets can therefore select
no passages. These parameters are declared defaults, not tuned to external outcomes.

Every POST has a `post-<persona>-evidence.json` trace with queries, rankings, selected
text/offsets/order, budget, status, and the exact initial POST context hash/size.
Selected passages are identical to the evidence in the request artifact. `EMPTY`
retrieval is reported explicitly and POST can proceed with its existing abstract and
ledger; `FAILED` retrieval prevents the POST call, records a null failed assessment,
and marks the run PARTIAL. No peer POST response enters queries or evidence selection.

## Issue history and independence

PRE sees the manuscript and only that reviewer's role/examples. It sees neither
peer PRE outputs nor the selection rationale. Every reviewer in a debate round
sees the same frozen complete structured history. Outputs commit only after the
round completes. POST uses the final debate snapshot and each reviewer's own PRE;
it cannot see same-stage peer POST judgments.

All specialists can assess all four novelty domains. Few-shot domain labels belong
to the particular vignette; there is no persona-to-domain mapping. One example or issue
can cover multiple domains. Unestablished domain judgments remain ABSTAIN.

The engine generates IDs. Each issue starts at version 1. A substantive revision
appends version 2, 3, etc. under the **same issue ID**, including revised statement,
evidence, severity and confidence. Original text and ratings stay unchanged in
the saved history. Arguments and response opportunities retain the version they
addressed. A reply to a historical argument stays attached to that version.
New arguments without a reply address the current version.

Only the originator can update an issue's stance. NARROW requires a revised
proposition; MAINTAIN/UNCERTAIN can also include a substantive revision.
RETRACT withdraws the concern. Peer concessions and silence do not retract,
validate, or resolve it. POST covers every own active issue exactly once.

New final-round/POST issues and revisions remain late. Earlier-version exposure
does not count as a response opportunity for a new version. Related and possible
duplicate concerns are linked without automatic merging or repeated-burden scoring.
The editor sees current propositions, POST judgments, constructive PRE fields,
and version-labeled arguments. Debate-call availability and the latest owner rationale
remain visible even when a POST response is missing. Historical objections are never automatically
reinserted as current concerns.

## Outputs and failures

Each invocation creates a unique run directory under the selected output root
(default: `results/exp-12/`; also configurable with `PEER_REVIEW_OUTPUT_DIR`):

- `result.json`: validated canonical assessments, all issue versions/updates,
  arguments, links, rounds, model-call provenance, and editor result.
- `report.md`: editor rationale/letter, individual PRE/POST judgments, current
  issues, diagnostics and usage.
- `tokens.json` and `calls.json`: all attempts, including failed/retried responses.
- Exact request/response artifacts, validated payloads, manuscript/abstract
  context, saved settings, frozen round snapshots, and stage checkpoints.
- `prompt-configuration.json`: all realized persona/stage system prompts, schemas,
  role profiles and examples; `run-metadata.json`: settings, effective models/rounds
  and retrieval backend/algorithm identity; per-persona POST evidence traces.

Code/prompt/schema/settings/manuscript hashes, available git commit, requested/sent settings,
context lengths/truncation and raw errors are retained. Model-reported settings
remain null when unavailable. Output artifacts contain manuscript text and reviews.

Successful zero-issue reviews have an empty issues array. Failed scholarly calls
have null payloads, never fabricated NONE novelty or a default verdict. Failed
PRE reviewers have skipped POST. Missing POST leaves final stance unavailable;
it does not erase the last debate state. Editor failure leaves no editor verdict.
A failed selection may use a recorded unweighted default panel. Run status is
COMPLETE, PARTIAL, or FAILED; the CLI exits nonzero for partial/failed runs.

The parser accepts bare JSON, JSON fences, and a balanced object inside brief
prose. Validation checks both schemas and graph references during retries.
Every retry retains the requested temperature; requested/sent values, retry kind,
attempt number, errors and usage are logged. HTTP 429/5xx errors are retryable; other HTTP errors
fail the call without repeatedly submitting the same rejected request.

Full manuscript text is used for PRE up to the declared limit. Round 1 also gets
that manuscript context; later rounds get the abstract and complete structured
history. A missing Abstract heading uses a labeled opening-text fallback.
Structured history is never sliced to fit: an over-limit request fails explicitly,
with its unsent context saved. Truncated PRE context marks the run PARTIAL.

Usage totals follow chat-completions conventions: input includes cached tokens,
output includes reasoning tokens. Missing usage/cache/reasoning remains null,
not zero. Costs remain unavailable without sufficient rates and usage; any known
subtotal is explicitly partial. Raw provider responses are kept for audit.

## Acceptance status

78 offline tests pass, covering the instruction audit and finalization changes recorded
in the changelog. The updated wheel, including retrieval, was built and installed in an
isolated temporary location; its installed command completed a 14-call fake-client workflow.
A live manuscript acceptance run still requires a chosen manuscript and
configured endpoint/model. No benchmark or downstream calibration is included.
Structural tests cannot establish the scholarly correctness of model judgments.
