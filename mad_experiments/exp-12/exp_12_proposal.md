# exp-12 changelog and remaining work

This file replaces the architecture proposal as the progress record. The filename is
retained so existing links keep working. The design specification remains
[exp_12_instructions.md](exp_12_instructions.md); later user decisions below take precedence.

## Decisions that supersede the original proposal

- Revisions retain the same issue ID. Preserve each earlier proposition, severity,
  confidence, and debate history; reassess severity and confidence when substance changes.
  Only the current version is current evidence. A revision needs its own response
  opportunities; earlier debate must not make a late revision look tested.
- No separate procedural-state enum. Use attributed originator stance and chronological events.
- Shared PRE/POST assessment fields; one proposition-confidence scale; no issue dimension.
- Model settings are stage-level, not persona-level.
- Keep progressing with review notifications between steps (user instruction, 2026-09-28).
  No separate approval is needed for each implementation step under that instruction.
- After implementation, consolidate to the main architecture, one config, one token tracker,
  one schema module, one README, installation metadata, and focused tests. No new proposal files.

## Completed before the current integration

- Checkpoint 1: validated schemas, stage settings, local connection configuration,
  token tracking, packaging metadata, and offline tests added in supporting files.
- Checkpoint 2: transactional ledger, engine-assigned IDs, frozen round snapshots,
  ownership/reference validation, and offline ledger tests added.
- Direct changes to exp-12.py:
  - Separated confidence (evidentiary support) from severity (damage conditional on truth).
  - Replaced severity anchors and specified reassessment after substantive revision.
  - Added the approved same-ID version-history rule to the core protocol.
  - Removed barrier_category from schemas, normalization, prompts, examples and reports.
  - Removed category multipliers, associated final-score caps, and category decision gates.
  - Verified syntax and 18 offline debate cases without credentials or model calls.
- Important limitation at this checkpoint: the original weighted scoring, old orchestration,
  and supporting helpers' different-ID narrowing were still present. Prompt changes alone
  did not implement the new runtime.

## Implemented — 2026-09-28

- Same-ID versioning is implemented in the schema and runtime, not just the prompt.
  Each revision stores its own statement, evidence, severity and confidence. Earlier
  issue records remain immutable. Arguments and response opportunities carry the version
  addressed. A historical reply remains historical after the issue changes.
- Revisions can accompany NARROW, MAINTAIN or UNCERTAIN; NARROW requires a revision.
  RETRACT withdraws the concern. Current views use the latest version and owner stance.
  Late revisions do not inherit earlier versions' testing opportunities.
- One CORE_REVIEW_PROTOCOL defines four-domain manuscript-supported novelty with
  abstention, insight, repair scope, severity, proposition confidence and qualitative verdicts.
  Removed global/novelty/insight/repair confidence, downward-bias novelty instructions,
  automatic LOW-to-REJECT gates, and the special novelty-counter path.
- Retained all ten role profiles. Added 30 schema-valid persona examples, including
  zero-issue successful reviews, bounded concerns and central failures. Severity,
  confidence, issue type, repair scope and verdict are not tied to fixed numerical templates.
- Replaced execution with selection → independent PRE → synchronous issue-centered debate
  → independent POST → qualitative editor synthesis. Default two-round success uses 14 calls.
  Selection has exactly three personas, rationale, and no authority weights.
- Removed all former scoring constants, reviewer-credibility updates, multipliers,
  penalties, aggregate scores, mandated decisions and score-dependent reports/CLI output.
- Strict generated schemas and graph validation run inside response retries. Invalid
  references, owner updates and missing POST coverage are rejected before mutation.
  Failed scholarly calls contain null payloads. Missing POST is not copied from PRE.
- Preserved complete round snapshots and plain-language statements. Added minimal abstract
  extraction, explicit truncation reporting, and context-limit failure without cutting issue arrays.
  Model manuscript content is supplied in user context, not a system-message prefix.
- Persisted canonical result, readable report, raw/validated responses, every attempt's
  usage/settings/errors, exact contexts, checkpoints, schema/prompt/code/manuscript hashes,
  and available commit metadata. Costs are optional and never inferred from stale prices.
- Consolidated the earlier helper package and config directory into exp-12.py, config.py,
  schemas.py and token_tracker.py. Removed duplicate YAML/config files, connection helper,
  ledger module and legacy import bridges after moving the needed behavior.
- Updated README and installation metadata. Runtime dependencies are pydantic and requests;
  installation includes the main command and its three flat helper modules.

## Verification completed

- 54 offline tests passed after the instruction-coverage audit: strict schemas, abstention, usage accounting, independent PRE/POST,
  same-round isolation, full-history carryover, version-specific argument targets,
  owner-only changes, late observations, missing stages, parsing/schema/graph retries,
  context-limit failures, and absence of architecture score/weight fields.
- All 30 persona examples validate against the actual PRE schema.
- Built and inspected co_econ_exp12-0.2.0 wheel; confirmed all runtime modules and command.
- Installed that wheel into an isolated temporary location, checked CLI help, and ran a
  complete 14-call workflow with an injected fake client from the installed script.
  Verified canonical result and report creation outside the source tree.
- Build tools were installed/upgraded only in the existing temporary test environment.

## Remaining acceptance step

A live manuscript run awaits the user's manuscript path and endpoint/model selection.
The request was sent during implementation. No benchmark rerun or downstream feature,
calibration, citation-retrieval, survey or publication work is included in this pass.

## Instruction-coverage audit — 2026-09-28

Audited the full specification against the actual executable path, schemas, config,
token tracking and tests. Architecture/prompt version is now exp12-v2.1/exp12-core-v2.1;
the persisted schema is unchanged. Five new regressions initially failed, reproducing
the gaps below; fixes plus three additional workflow checks bring the suite to 54 tests.

Corrections made during this audit:

- Current-issue views now deep-copy nested evidence/links. Mutating a returned view
  cannot alter immutable issue history.
- Editor context includes debate availability, including failed rounds with zero issues,
  and the most recent owner rationale even when POST is missing.
- Arguments explicitly indicate whether they address an active current issue.
  Retracted concerns remain historical in both editor context and the rendered report.
- Malformed optional token-detail objects no longer crash a completed call or discard
  valid input/output totals; missing optional details remain unknown.
- Abstract extraction stops at a section heading at EOF, even without a final newline.

### Whole-document coverage

| Specification | Coverage and implementation |
|---|---|
| §§0–3: purpose, raw-record separation, invariants | Implemented in run_review, Ledger and RunResult. No internal paper score or persona authority. Research motivation is preserved in the source instructions. |
| §§4–5: constructs and contracts | CORE_REVIEW_PROTOCOL plus schemas.py: four novelty domains, insight, repair, verdict, strengths, revision path, issue damage/support and plain-language statements. |
| §6: identity, taxonomy, chronology | Ledger validates IDs/ownership/visible references, appends versions, retains attributed duplicate links and derives late/response-opportunity facts. Same-ID revisions follow the later approval. |
| §7: stage flow | run_review collects independent PRE, synchronous round outputs, independent POST and qualitative editor synthesis. |
| §8: no authority weighting | SelectionOutput has only personas/rationale; no credibility or weighting path exists. |
| §9: manuscript context | extract_abstract supplies heading-based extraction or labeled opening fallback. No broad role-window machinery or extra summary call. |
| §§10–12: prompts, examples, novelty discipline | One core protocol, five stage prompts and 30 validated persona examples. No automatic LOW rejection or instruction to err lower. |
| §13: model configuration | config.py defaults to reasoning off, temperatures 0 except debate 0.35. ReviewerCalls records requested/sent temperatures and retry changes. No causal claim about earlier reasoning experiments is made in runtime. |
| §14: parsing, schema validation, retries | Tolerant extraction followed by strict Pydantic and graph validation; all attempts retained. Null failure payloads, explicit selection fallback. |
| §15: deletions | Old scoring, weighting, penalties, novelty-counter path, global judgment confidence, broad section machinery and legacy aliases are absent from runtime. |
| §§16–17: retained information | Raw PRE/POST judgments, evidence, versioned issues, actions, links and timing remain available for later feature construction. |
| §§18–24: downstream research | Intentionally not implemented here: feature formulas, calibration, generalist comparisons, surveys, empirical validation and publication framing. These are research plans, not missing runtime functions; §26 explicitly defers them. |
| §25: observability | RunResult, exact request/response files, settings, checkpoints, round metrics, token/cost records and hashes. Unknown provider metadata stays null. |
| §26: non-goals | No calibration, citation retrieval, benchmark expansion, survey construction or persona-library redesign. |
| §27: workflow | Design review, implementation and offline checks completed. One-paper live acceptance is still pending; benchmark rerun remains deferred. |
| §28: acceptance criteria | Detailed mapping below; structural/runtime criteria tested, semantic model behavior still needs live/human review. |
| §29: implementation choices | Resolved by approved scales/contracts and later same-ID versioning. Issue dimension and procedural enum were explicitly removed by user-approved simplifications. |
| §30: target pipeline | Implemented end to end in run_review. |

### Acceptance criteria (§28)

The original list skips 14 and repeats 15; both 15 entries are represented below.
Test names are in tests/test_checkpoint1.py and tests/test_ledger.py. These establish
mechanical behavior, not the factual correctness of a model-generated review.

| Item | Implementation / verification |
|---|---|
| 1: exactly three, no authority weights | SelectionOutput; selection contract tests. |
| 2: independent PRE | Separate requests, no panel rationale/peer PRE; full-workflow independence test. |
| 3: zero issues is valid | Empty IssueCandidate list; success-vs-failure tests. |
| 4–5: engine-generated unique IDs | Ledger._create_issue/_append_response; deterministic ID and rollback tests. |
| 6–7: existing issue targets, never paper | ArgumentCandidate ID pattern and validate_response; invalid/same-round/cross-issue reference tests. |
| 8: no same-round peer visibility | Frozen Snapshot plus collect-then-commit; identical-context workflow tests. |
| 9: accumulated structured history | Ledger._begin carries all records; multi-round full-history test includes more than four arguments. |
| 10: plain-language issues reach debate | IssueCandidate and complete snapshots; explicit snapshot/workflow assertions. |
| 11: final-round concerns stay late | issue_views; new-issue and revision tests verify zero inherited response opportunities. |
| 12: POST observations remain untested | late_observations, revision phase and late flags; POST tests and editor instructions. Whether prose obeys this needs model review. |
| 13: grounded issues/consequential omissions | Required nonempty Evidence records and explicit CORE instructions. Structural tests can reject missing evidence, not establish its truth. |
| 15 (severity): conditional issue damage, no jumps | Severity 1–10 plus rationale; continuous scale validation and absence of scoring machinery. |
| 15 (confidence): initial issue support | Required initial_confidence; bounded numeric validation. |
| 16: claim-bearing argument confidence | ArgumentCandidate validates CHALLENGE/DEFENSE confidence and evidence. |
| 17: questions/concessions have no confidence | ArgumentCandidate requires null for those actions; contract tests. |
| 18–20: no global/insight/repair confidence | Closed Assessment/DomainJudgment/RepairScope schemas; generated-schema checks. |
| 21–22: ABSTAIN distinct from NONE | Tagged judgment schemas; explicit distinction tests. |
| 23: comparable PRE/POST retained | Shared Assessment contract; RunResult and workflow persistence assertions. |
| 24: append-only version history | Separate Issue/IssueRevision/IssueUpdate records; mutation-isolation, version-target and historical-reply tests. |
| 25–26: no dynamic credibility or weights | No runtime credibility/aggregation functions; closed schemas and result-field checks. |
| 27: no barrier_category | Removed from runtime contracts, prompts and output; forbidden-field tests. |
| 28–31: no final score, penalties, novelty caps or mixture | Removed runtime; schema/result checks and source audit. Token costs and operational limits are not paper-quality scores. |
| 32: no mechanical resurrection of withdrawn/old concerns | Latest active version only; historical arguments labeled; retracted-report/editor regression test. |
| 33: current ledger + POST drives editor | editor_context prioritizes current views and POST, retaining PRE strengths, debate failures and owner reasons; context tests. |
| 34: models/settings/retries logged | CallAttempt, exact requests/raw replies and TokenTracker; full-workflow retry/usage tests. |
| 35: downstream features remain derivable | RunResult retains all independent judgments, versions, arguments, links, opportunities and provenance; persistence round-trip tests. |

No outstanding implementation gap was found after these corrections within the
architecture scope. Live provider compatibility, substantive evidence accuracy,
calibration of judgments and quality of author feedback are not established by
offline fixtures and remain acceptance/evaluation work.

## Validation and live-run status

No live model calls have been made during implementation. Offline validation requires no
API key. Live calls require the user's chosen endpoint/model and any credentials that
endpoint requires; keyless local endpoints are supported. No IDE authentication file was accessed.
