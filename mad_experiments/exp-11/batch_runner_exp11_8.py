#!/usr/bin/env python3
"""
Manifest-driven batch runner for the exp-11-8 multi-agent peer-review engine.

Adapted from batch_runner_exp8.py. Differences from the exp-8 runner:
  * Driven by a CSV manifest (paper_manifest_exp11_8.csv) rather than a directory
    glob. Only rows whose ``already_run`` column is NOT "yes" are processed; rows
    already run are handled by a separate adapter and are skipped here.
  * Imports the hyphenated engine file (exp-11-8.py) via importlib.
  * Extracts text from PDFs at runtime (most manifest papers are PDFs; only the
    FEDS/IFDP-2020 rows point at pre-extracted .txt). The engine's own CLI only
    reads .txt, so PDF extraction lives here (PyMuPDF, pdfplumber fallback).
  * Maps the exp-11 TokenTracker.get_summary() schema (input/output/cache_read/
    cache_creation/total + total_cost) into SEPARATED token columns in
    batch_summary.csv. The old exp-8 keys (total_input_tokens / estimated_cost_usd)
    do NOT exist in exp-11's schema and are intentionally not used.

SCHEMA-MAPPING CAVEAT (Step 6 — do not silently drop fields):
    exp-11-8 audits emit ``verdict`` in {ACCEPT, RESUBMIT, REJECT} and use
    ``severity_score`` / ``barrier_category`` (there is NO ``severity_delta``).
    Downstream calibration historically expects ``verdict`` in {PASS, REVISE, FAIL}
    and a ``severity_delta`` field. This runner writes the engine's NATIVE fields
    VERBATIM into structured_results.json (lossless). It does NOT perform the
    PASS/REVISE/FAIL or severity_delta normalization — that is the calibration
    step's responsibility and is out of scope here. A future reader adapting
    calibration must map ACCEPT->PASS, RESUBMIT->REVISE, REJECT->FAIL and derive a
    severity_delta from severity_score/barrier_category. Nothing is lost on disk.

Usage:
    python3 batch_runner_exp11_8.py --limit 1 --no-confirm --run-id smoke_test
"""
import argparse
import asyncio
import csv
import importlib.util
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# This repository keeps the complete experiment under benchmark/.
BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
CONFIG_DIR = BENCHMARK_DIR / "config"
PAPERS_DIR = BENCHMARK_DIR / "papers"

ENGINE_DIR = BENCHMARK_DIR
ENGINE_FILE = ENGINE_DIR / "exp-11-8.py"

DEFAULT_MANIFEST = BENCHMARK_DIR / "paper-manifest.csv"
RESULTS_ROOT = REPO_ROOT / "results" / "batch_experiments" / "exp11_8"

# Cross-check ground truth (manifest tier is authoritative; this is a fallback only).
GROUND_TRUTH_CSV = BENCHMARK_DIR / "ground_truth_2019_2020.csv"

# --------------------------------------------------------------------------
# Engine import (hyphenated filename cannot be imported normally)
# --------------------------------------------------------------------------
def load_engine():
    """Import exp-11-8.py, ensuring its local imports resolve."""
    # Put the engine and config dirs on sys.path so local imports resolve.
    sys.path.insert(0, str(ENGINE_DIR))
    sys.path.insert(0, str(CONFIG_DIR))
    spec = importlib.util.spec_from_file_location("exp_11_8", str(ENGINE_FILE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


engine = None

# The engine writes its artifacts to a MODULE-GLOBAL OUTPUT_DIR (set via
# set_output_dir) that is read throughout a run (per-persona debug JSONs, the
# final .md, token_usage_*.json). Concurrent engine invocations would therefore
# interleave/clobber each other's artifact destinations. We guard the entire
# (set_output_dir -> run_peer_review_system) critical section with a lock so the
# engine's global state stays consistent. The --max-parallel semaphore still
# governs scheduling and the (CPU-bound) PDF extraction runs outside the lock.
_ENGINE_LOCK = asyncio.Lock()


# --------------------------------------------------------------------------
# Manifest + ground truth
# --------------------------------------------------------------------------
def load_manifest(manifest_path: Path):
    """Return list of dict rows from the manifest CSV."""
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def new_papers(rows):
    """Rows whose already_run column is NOT 'yes' (the papers we must process)."""
    out = []
    for row in rows:
        already = (row.get("already_run") or "").strip().lower()
        if already != "yes":
            out.append(row)
    return out


def resolve_source_path(row, manifest_path: Path) -> Path:
    """Resolve portable paths and repair stale manifest paths by doc_id.

    The canonical manifest uses paths relative to benchmark/. Older manifests
    in this repository contain paths from other machines and occasionally name
    a PDF where the local source is a TXT file.
    """
    raw = (row.get("source_path") or "").strip()
    candidates = []
    if raw:
        given = Path(raw).expanduser()
        candidates.extend([
            given,
            manifest_path.parent / given,
            BENCHMARK_DIR / given,
        ])
    doc_id = (row.get("doc_id") or "").strip()
    if doc_id:
        candidates.extend([PAPERS_DIR / f"{doc_id}.txt", PAPERS_DIR / f"{doc_id}.pdf"])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No local source found for {doc_id}; checked: {rendered}")


def load_ground_truth_fallback():
    """doc_id -> tier crosscheck map (deduped). Manifest tier is authoritative."""
    gt = {}
    if not GROUND_TRUTH_CSV.exists():
        return gt
    with open(GROUND_TRUTH_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            doc_id = (row.get("doc_id") or "").strip()
            tier = (row.get("Tier") or row.get("tier") or "").strip()
            if doc_id and doc_id not in gt:
                gt[doc_id] = tier
    return gt


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF. PyMuPDF primary (matches how exp-11-8 inputs were
    prepared), pdfplumber fallback."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        text = "".join(page.get_text() for page in doc)
        doc.close()
        if text.strip():
            return text
        print(f"  [WARN] PyMuPDF returned empty text for {pdf_path.name}; trying pdfplumber")
    except Exception as exc:
        print(f"  [WARN] PyMuPDF extraction failed for {pdf_path.name}: {exc}; trying pdfplumber")

    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as pdf:
        return "".join((page.extract_text() or "") for page in pdf.pages)


def load_paper_text(source_path: Path) -> str:
    if source_path.suffix.lower() == ".txt":
        with open(source_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return extract_text_from_pdf(source_path)


# --------------------------------------------------------------------------
# Per-paper processing
# --------------------------------------------------------------------------
async def process_single_paper(
    row, paper_num, total, run_id, rounds, gt_fallback, semaphore, manifest_path, resume
):
    """Extract, run the engine, and write per-paper outputs matching exp-8 shape."""
    async with semaphore:
        doc_id = (row.get("doc_id") or "").strip()
        tier = (row.get("tier") or "").strip()  # manifest tier is authoritative
        year = (row.get("year") or "").strip()
        series = (row.get("series") or "").strip()
        source_path = resolve_source_path(row, manifest_path)

        # Cross-check against the ground-truth CSV (informational only).
        gt_cross = gt_fallback.get(doc_id)
        if gt_cross and tier and gt_cross != tier:
            print(f"  [NOTE] {doc_id}: manifest tier {tier} != ground_truth CSV {gt_cross} "
                  f"(using manifest tier)")

        print(f"\n{'=' * 80}")
        print(f"Processing paper {paper_num}/{total}: {doc_id}")
        print(f"Ground truth tier: {tier} | year: {year} | series: {series}")
        print(f"Source: {source_path}")
        print(f"{'=' * 80}\n")

        start_time = time.time()
        paper_output_dir = RESULTS_ROOT / run_id / doc_id
        completed_result = paper_output_dir / "structured_results.json"
        if resume and completed_result.exists():
            print(f"\n[SKIP] {doc_id}: completed output already exists at {completed_result}")
            return {"success": True, "doc_id": doc_id, "skipped": True}
        paper_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            paper_text = load_paper_text(source_path)
            char_count = len(paper_text)
            word_count = len(paper_text.split())
            print(f"Paper length: {char_count} chars, {word_count} words")
            if char_count == 0:
                raise ValueError("Extracted paper text is empty.")

            # --- Engine critical section (global OUTPUT_DIR) — serialized ---
            async with _ENGINE_LOCK:
                engine.set_output_dir(str(paper_output_dir))
                results = await engine.run_peer_review_system(
                    paper_text=paper_text,
                    human_directive="Perform a rigorous academic audit.",
                    rounds=rounds,
                )

            duration = time.time() - start_time

            final_score = results.get("final_score")
            decision = results.get("decision")
            token_usage = results.get("token_usage", {})  # == TokenTracker.get_summary()
            output_file = results.get("output_file")
            if str(decision).startswith("SYSTEM ERROR"):
                raise RuntimeError(f"Engine did not produce valid audits: {decision}")

            # Ensure the engine's report lands as peer_review_report.md (exp-8 layout).
            # The engine already wrote its peer_review_*.md into paper_output_dir; copy
            # to the canonical name expected by downstream tooling.
            if output_file and Path(output_file).exists():
                dest_report = paper_output_dir / "peer_review_report.md"
                if Path(output_file).resolve() != dest_report.resolve():
                    shutil.copy(output_file, dest_report)

            metadata = {
                "doc_id": doc_id,
                "paper_path": str(source_path),
                "ground_truth_tier": tier,
                "year": year,
                "series": series,
                "final_score": final_score,
                "decision": decision,
                "duration_seconds": duration,
                "char_count": char_count,
                "word_count": word_count,
                "token_usage": token_usage,
                "timestamp": datetime.now().isoformat(),
                "engine": "exp-11-8",
            }
            with open(paper_output_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # structured_results.json — matches batch_runner_exp8.py key layout so
            # calibration can read it. Native exp-11 fields are written VERBATIM
            # (see SCHEMA-MAPPING CAVEAT at top): independent_audits carry
            # verdict in {ACCEPT,RESUBMIT,REJECT} and severity_score/barrier_category.
            structured_results = {
                "metadata": metadata,
                "independent_audits": results.get("independent_audits"),
                "debate_history": results.get("debate_history"),
                "final_report": results.get("final_report"),
                "audit_trail": results.get("audit_trail"),
                # Extra (lossless) — exp-11 also returns scoring components; harmless
                # for calibration (which reads specific keys) and useful for analysis.
                "score_components": results.get("score_components"),
            }
            with open(paper_output_dir / "structured_results.json", "w", encoding="utf-8") as f:
                json.dump(structured_results, f, indent=2)

            print(f"\n[OK] {doc_id}: score={final_score}, decision={decision}, {duration:.1f}s")
            print(f"     Output: {paper_output_dir}")

            return {"success": True, "doc_id": doc_id, "metadata": metadata}

        except Exception as exc:
            duration = time.time() - start_time
            print(f"\n[ERROR] {doc_id}: {exc}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "doc_id": doc_id,
                "error": str(exc),
                "duration": duration,
                "tier": tier,
                "year": year,
                "series": series,
            }


# --------------------------------------------------------------------------
# batch_summary.csv (SEPARATED token columns)
# --------------------------------------------------------------------------
SUMMARY_COLUMNS = [
    "doc_id", "tier", "year", "series", "final_score", "decision", "duration_seconds",
    "input_tokens", "output_tokens", "thinking_tokens", "cache_read_tokens", "cache_creation_tokens",
    "total_tokens", "estimated_cost",
]


def _row_from_result(result):
    if result.get("success"):
        m = result["metadata"]
        tu = m.get("token_usage", {}) or {}
        # VERIFIED against token_tracker.py TokenTracker.get_summary():
        #   summary["total_tokens"] = {"input","output","total","cache_read","cache_creation"}
        #   summary["total_cost"]   = float
        tot = tu.get("total_tokens", {}) or {}
        return {
            "doc_id": m["doc_id"],
            "tier": m["ground_truth_tier"],
            "year": m.get("year", ""),
            "series": m.get("series", ""),
            "final_score": m["final_score"],
            "decision": m["decision"],
            "duration_seconds": m["duration_seconds"],
            "input_tokens": tot.get("input", 0),
            "output_tokens": tot.get("output", 0),
            "thinking_tokens": tot.get("thinking", 0),
            "cache_read_tokens": tot.get("cache_read", 0),
            "cache_creation_tokens": tot.get("cache_creation", 0),
            "total_tokens": tot.get("total", 0),
            "estimated_cost": tu.get("total_cost", 0.0),
        }
    return {
        "doc_id": result["doc_id"],
        "tier": result.get("tier", ""),
        "year": result.get("year", ""),
        "series": result.get("series", ""),
        "final_score": None,
        "decision": "ERROR",
        "duration_seconds": result.get("duration", 0),
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "total_tokens": 0,
        "estimated_cost": 0.0,
    }


def write_summary_csv(summary_path: Path, results):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Append if it already exists (resume-friendly); write header only when new.
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for result in results:
            writer.writerow(_row_from_result(result))


# --------------------------------------------------------------------------
# Batch orchestration
# --------------------------------------------------------------------------
async def run_batch(args):
    global engine
    print("\n" + "=" * 80)
    print("BATCH RUNNER — exp-11-8 (manifest-driven)")
    print("=" * 80)

    manifest_path = Path(args.manifest).expanduser().resolve()
    print(f"\nManifest: {manifest_path}")
    print(f"Run ID:   {args.run_id}")
    print(f"Rounds:   {args.rounds}")

    rows = load_manifest(manifest_path)
    pending = new_papers(rows)
    total_new = len(pending)
    print(f"\nManifest rows: {len(rows)} | new (already_run != yes): {total_new}")

    if args.doc_id:
        pending = [row for row in pending if (row.get("doc_id") or "").strip() == args.doc_id]
        if not pending:
            raise SystemExit(f"doc_id not found among pending manifest rows: {args.doc_id}")
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"Papers to process this run: {len(pending)}")

    if not pending:
        print("Nothing to process. Exiting.")
        return 0

    validation_errors = []
    for row in pending:
        try:
            resolve_source_path(row, manifest_path)
        except FileNotFoundError as exc:
            validation_errors.append(str(exc))
    if validation_errors:
        print("\nManifest validation failed:")
        for error in validation_errors:
            print(f"  - {error}")
        raise SystemExit(2)
    print(f"Validated {len(pending)} paper source paths.")

    if args.validate_only:
        print("Validation-only mode complete; no API calls were made.")
        return 0

    engine = load_engine()

    gt_fallback = load_ground_truth_fallback()

    print("\nPlanned papers:")
    for i, row in enumerate(pending[:10], 1):
        print(f"  {i}. {row['doc_id']} (tier {row['tier']}, {row['series']}-{row['year']})")
    if len(pending) > 10:
        print(f"  ... and {len(pending) - 10} more")

    print("\nCost varies with prompt/output tokens; measured usage is written after each completed paper.")
    print(f"Max parallel scheduling: {args.max_parallel} "
          f"(engine runs are serialized due to global OUTPUT_DIR state)")
    print(f"Output dir: {RESULTS_ROOT / args.run_id}")

    if not args.no_confirm:
        response = input("\nProceed with these API calls? (yes/no): ").strip().lower()
        if response not in ("yes", "y"):
            print("Aborted.")
            return

    batch_start = time.time()
    semaphore = asyncio.Semaphore(max(1, args.max_parallel))
    tasks = [
        process_single_paper(
            row, i, len(pending), args.run_id, args.rounds, gt_fallback,
            semaphore, manifest_path, args.resume
        )
        for i, row in enumerate(pending, 1)
    ]
    results = await asyncio.gather(*tasks)
    batch_duration = time.time() - batch_start

    summary_csv = RESULTS_ROOT / args.run_id / "batch_summary.csv"
    skipped = [r for r in results if r.get("skipped")]
    completed_results = [r for r in results if not r.get("skipped")]
    write_summary_csv(summary_csv, completed_results)
    successful = [r for r in completed_results if r.get("success")]
    failed = [r for r in completed_results if not r.get("success")]

    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)
    print(
        f"Scheduled: {len(results)} | Successful: {len(successful)} | "
        f"Resumed/skipped: {len(skipped)} | Failed: {len(failed)}"
    )
    print(f"Total time: {batch_duration / 60:.1f} min")
    if successful:
        total_cost = sum(
            (r["metadata"].get("token_usage", {}) or {}).get("total_cost", 0.0) for r in successful
        )
        print(f"Total measured cost: ${total_cost:.4f}")
    if failed:
        print("\nFailed papers:")
        for r in failed:
            print(f"  - {r['doc_id']}: {r.get('error', 'unknown')}")
    print(f"\nSummary CSV: {summary_csv}")
    print(f"Outputs:     {RESULTS_ROOT / args.run_id}")
    print("=" * 80 + "\n")
    return 1 if failed else 0


def parse_args():
    p = argparse.ArgumentParser(description="Manifest-driven batch runner for exp-11-8.")
    p.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST),
                   help="Path to the paper manifest CSV.")
    p.add_argument("--run-id", type=str, default="exp11_8_batch",
                   help="Run identifier (subfolder name). Pass your own; no timestamps are auto-generated.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N NEW papers.")
    p.add_argument("--doc-id", type=str, default=None,
                   help="Process one exact pending doc_id (useful for smoke tests).")
    p.add_argument("--max-parallel", type=int, default=3,
                   help="Max papers scheduled in parallel (engine runs still serialize on global state).")
    p.add_argument("--rounds", type=int, default=2, help="Number of debate rounds.")
    p.add_argument("--no-confirm", action="store_true",
                   help="Skip the pre-run confirmation prompt.")
    p.add_argument("--validate-only", action="store_true",
                   help="Validate the manifest and all source paths without loading credentials or calling the API.")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip papers that already have structured_results.json (default: true).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        sys.exit(asyncio.run(run_batch(args)))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Partial results may be saved.")
    except Exception as exc:
        print(f"\n\nFATAL ERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
