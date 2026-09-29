# exp-12: specialist peer review

Select three specialists → independent PRE reviews → synchronous debate → independent
POST reassessment → qualitative editor synthesis. The architecture preserves raw
judgments and issue history without computing paper-quality scores or reviewer weights.

## Layout

```text
exp-12.py                         Prompts, examples, ledger, workflow and CLI
config.py                        Defaults and optional JSON overrides
schemas.py                       Review and saved-record contracts
token_tracker.py                 Usage and optional cost accounting
retrieval.py                     Deterministic manuscript evidence retrieval
pyproject.toml                   Installation and dependencies
tests/                           Offline contract, retrieval and workflow tests
analysis/                        Downstream discussion stubs only
    architecture_calibration.py  Architecture comparisons and paper-level resampling
    feature_selection.py         Candidate features, ablation and stability
    evaluation_analysis.py       External validity, debate gains and baselines
```

The three `analysis/` files document expected inputs, outputs and unresolved TODOs;
their functions raise `NotImplementedError`. Metrics, formulas, weights and statistical
specifications remain undecided. Future splitting/resampling uses papers as independent
units, with outcomes and PRE/POST/mechanism measures kept distinguishable.

See [the original specification](exp_12_instructions.md) and
[the changelog](exp_12_proposal.md) for detailed decisions and validation history.

## Install and test

From this folder, using Python 3.9 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m unittest discover -s tests -v
```

Dependencies are `pydantic` and `requests`; `requirements.txt` provides an equivalent
installation shortcut. Tests run offline without credentials. Installation includes
the review command and four helper modules; `analysis/` remains discussion scaffolding.

## Run

Supply a UTF-8 manuscript text file and a chat-completions-compatible endpoint/model:

```sh
export PEER_REVIEW_API_BASE="https://your-provider.example/v1"
export PEER_REVIEW_MODEL="your-model-id"
# Set PEER_REVIEW_API_KEY if your endpoint requires it.
python exp-12.py /path/to/manuscript.txt --output-dir /path/to/reviews
```

Keyless local endpoints are supported. `OPENAI_BASE_URL` and `OPENAI_API_KEY` are
accepted fallbacks. Choose a model supported by your endpoint.

Defaults: two debate rounds, 14 successful review calls, reasoning off, temperature
0 except debate at 0.35. Retries retain the requested temperature. Use `--rounds`,
`--model`, or `--config settings.json` to override [configuration defaults](config.py);
explicit stage models take precedence over `--model`.

## Context and issue history

- PRE is independent; each debate round shares one frozen ledger snapshot. POST
  sees the final debate state and its own PRE, with no peer POST responses.
- All personas can assess all four novelty domains. Domain labels belong to claims.
- Substantive revisions retain the issue ID and append statement, evidence, severity
  and confidence versions. Arguments retain the version they address. Silence does
  not resolve an issue; late revisions do not inherit earlier response opportunities.
- PRE and round 1 use bounded manuscript text; later rounds use the abstract and
  complete ledger. Oversized contexts fail explicitly without dropping ledger entries.

POST defaults to retrieved BM25 evidence: at most 12,000 serialized characters,
with raw passages capped at 25% of manuscript length. Exact text, offsets, queries
and rankings are saved. `post_manuscript_context` also supports `abstract_only` and
bounded `full_manuscript`. Embedding/hybrid retrieval requires an explicit backend
passed to `run_review`; no embedding model is downloaded automatically.

## Outputs and status

Each run creates a unique folder under `--output-dir` (default `results/exp-12/`):

- `result.json` and `report.md`: raw assessments/history and readable editor review.
- `calls.json` and `tokens.json`: attempts, errors, usage and optional priced costs.
- Exact requests/responses, settings, prompt configuration, snapshots, POST evidence
  traces and reproducibility hashes.

Failed assessments remain null; successful empty issue lists remain valid. Empty
retrieval is recorded; failed retrieval prevents that POST call. Runs report
`COMPLETE`, `PARTIAL` or `FAILED`; the CLI exits nonzero for incomplete runs.
Missing usage or prices remain unknown.

Recorded validation: 78 offline tests and an installed-package workflow passed.
Live manuscript acceptance is pending. Downstream analyses are unimplemented.
