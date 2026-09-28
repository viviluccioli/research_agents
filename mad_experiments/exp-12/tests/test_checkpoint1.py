"""Offline tests of the approved helper contracts; no imports of exp-12.py."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from config import ModelConfig
from schemas import (
    ArgumentCandidate, CallAttempt, DebateOutput, Issue, IssueCandidate,
    LinkProposal, OriginatorUpdate, Outcome, PostAssessment, PreAssessment,
    RunResult, SelectionOutput, TokenUsage, MODEL_OUTPUT_SCHEMAS,
)
from token_tracker import TokenTracker

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = {"kind": "MANUSCRIPT", "locator": "Table 2", "support": "The estimated gap increases."}


def candidate():
    return {
        "issue_type": "substantive_flaw", "novelty_domains": [],
        "technical_statement": "Group-specific estimates contradict the claimed narrowing of the gap.",
        "plain_language_statement": "The paper says the gap shrank, but its table shows it grew.",
        "severity": 7.2, "severity_rationale": "This contradicts the central distributional claim.",
        "initial_confidence": 8.0, "evidence": [deepcopy(EVIDENCE)], "related_issue_ids": [],
    }


def core():
    return {
        "novelty_by_domain": {d: {"level": "ABSTAIN", "reason": "Insufficient positioning evidence."}
                              for d in ("methodological", "empirical", "theoretical", "policy")},
        "insight": {"level": "MEDIUM", "rationale": "Useful economic interpretation.", "evidence": [EVIDENCE]},
        "repair_scope": {"status": "ASSESSED", "level": 1, "rationale": "Narrow the interpretation."},
        "revision_path": "Reconcile the distributional claim with Table 2.",
        "verdict": "RESUBMIT", "assessment_rationale": "The central framing needs revision.",
    }


def pre():
    return {**core(), "structural_strength": "Clear estimand.",
            "best_case_contribution": "Useful subgroup comparison.", "issues": []}


def attempt(**overrides):
    data = {
        "call_id": "call-1", "stage": "PRE", "persona": "Econometrician",
        "round_number": 0, "attempt": 1, "retry_kind": "INITIAL",
        "requested_model": "fixture-model", "reported_model": None,
        "requested_temperature": 0.0, "sent_temperature": 0.0,
        "reported_temperature": None, "reasoning_requested": "none", "reasoning_sent": "none",
        "request_context": None, "raw_response_path": None,
        "status": "OK", "errors": [], "started_at": "2026-09-25T12:00:00+00:00",
        "elapsed_seconds": 0.1,
        "usage": {"input_tokens": 100, "output_tokens": 40, "reasoning_tokens": 10,
                  "cache_read_tokens": 20, "cache_write_tokens": 10},
    }
    data.update(overrides)
    return CallAttempt.model_validate(data)


class SchemaTests(unittest.TestCase):
    def test_selection_requires_three_distinct_known_personas_without_weights(self):
        valid = {"selected_personas": ["Econometrician", "Theorist", "Historian"],
                 "selection_rationale": "Complementary risks."}
        SelectionOutput.model_validate(valid)
        for names in (["Theorist"] * 3, ["Theorist"], ["Theorist", "Historian", "Unknown"]):
            with self.subTest(names=names), self.assertRaises(ValidationError):
                SelectionOutput.model_validate({**valid, "selected_personas": names})
        with self.assertRaises(ValidationError):
            SelectionOutput.model_validate({**valid, "weights": {"Theorist": 1}})

    def test_zero_issues_is_success_not_failure(self):
        result = PreAssessment.model_validate(pre())
        good = Outcome[PreAssessment](status="OK", payload=result, call_ids=["c1"], reason=None)
        failed = Outcome[PreAssessment](status="FAILED", payload=None, call_ids=["c2"], reason="Parse failure")
        self.assertEqual(good.payload.issues, [])
        self.assertIsNone(failed.payload)
        with self.assertRaises(ValidationError):
            Outcome[PreAssessment](status="FALLBACK", payload=result, call_ids=[], reason="Failed")
        with self.assertRaises(ValidationError):
            Outcome[PreAssessment](status="FAILED", payload=result, call_ids=[], reason="Failed")

    def test_same_constructs_in_pre_and_post_with_distinct_abstention(self):
        first = PreAssessment.model_validate(pre())
        last = PostAssessment.model_validate({**core(), "originator_updates": [], "late_observations": []})
        self.assertEqual(first.model_dump(include=set(core())), last.model_dump(include=set(core())))
        raw = pre()
        raw["novelty_by_domain"]["policy"] = {"level": "NONE", "rationale": "No policy contribution claimed.", "evidence": [EVIDENCE]}
        result = PreAssessment.model_validate(raw)
        self.assertEqual(result.novelty_by_domain.policy.level, "NONE")
        self.assertEqual(result.novelty_by_domain.theoretical.level, "ABSTAIN")
        raw["repair_scope"] = {"status": "ABSTAIN", "reason": "Cannot assess feasible repair."}
        raw["insight"] = {"level": "ABSTAIN", "reason": "Insufficient evidence."}
        PreAssessment.model_validate(raw)

    def test_pre_cannot_reference_unseen_issues(self):
        raw = pre()
        raw["issues"] = [{**candidate(), "related_issue_ids": ["I000001"]}]
        with self.assertRaises(ValidationError):
            PreAssessment.model_validate(raw)

    def test_scales_reject_coercion_nonfinite_and_out_of_range(self):
        for key, values in {"severity": [0, 11, "7", True, float("nan"), float("inf")],
                            "initial_confidence": [-1, 11, "8", True]}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                    IssueCandidate.model_validate({**candidate(), key: value})
        IssueCandidate.model_validate({**candidate(), "initial_confidence": 0.0})
        for value in [True, 1.5, "1", 5]:
            raw = pre()
            raw["repair_scope"]["level"] = value
            with self.subTest(repair=value), self.assertRaises(ValidationError):
                PreAssessment.model_validate(raw)

    def test_model_cannot_generate_ids_or_obsolete_fields(self):
        for key, value in {"issue_id": "I000001", "dimension": "soundness", "procedural_state": "OPEN", "severity_score": 7}.items():
            with self.subTest(key=key), self.assertRaises(ValidationError):
                IssueCandidate.model_validate({**candidate(), key: value})
        for value in ["no_issue", "unknown"]:
            with self.assertRaises(ValidationError):
                IssueCandidate.model_validate({**candidate(), "issue_type": value})

    def test_argument_confidence_and_evidence_are_action_specific(self):
        raw = {"issue_id": "I000001", "responds_to_argument_id": None,
               "content": "The table supports the concern.", "evidence": [EVIDENCE], "confidence": 8.0}
        for action in ("CHALLENGE", "DEFENSE"):
            ArgumentCandidate.model_validate({**raw, "action": action})
            with self.assertRaises(ValidationError):
                ArgumentCandidate.model_validate({**raw, "action": action, "confidence": None})
            with self.assertRaises(ValidationError):
                ArgumentCandidate.model_validate({**raw, "action": action, "evidence": []})
        for action in ("QUESTION", "CONCESSION"):
            ArgumentCandidate.model_validate({**raw, "action": action, "confidence": None, "evidence": []})
            with self.assertRaises(ValidationError):
                ArgumentCandidate.model_validate({**raw, "action": action})
        with self.assertRaises(ValidationError):
            ArgumentCandidate.model_validate({**raw, "action": "DEFENSE", "issue_id": "paper"})

    def test_narrowing_contract_and_duplicate_updates(self):
        raw = {"issue_id": "I000001", "stance": "NARROW", "rationale": "Only the subgroup claim fails.",
               "confidence": 8.0, "revised_issue": candidate()}
        narrowed = OriginatorUpdate.model_validate(raw)
        with self.assertRaises(ValidationError):
            OriginatorUpdate.model_validate({**raw, "revised_issue": None})
        with self.assertRaises(ValidationError):
            OriginatorUpdate.model_validate({**raw, "confidence": 3.0})
        with self.assertRaises(ValidationError):
            DebateOutput(arguments=[], new_issues=[], issue_links=[], originator_updates=[narrowed, narrowed])
        with self.assertRaises(ValidationError):
            LinkProposal(issue_ids=["I000001", "I000001"], relationship="RELATED", rationale="Duplicate")

    def test_engine_issue_preserves_provenance_without_redundant_state(self):
        raw = {**candidate(), "source_call_id": "c1", "event_sequence": 1, "issue_id": "I000001",
               "originator": "Perspective", "phase_created": "PRE", "round_created": 0,
               }
        Issue.model_validate(raw)
        for addition in ({"round_created": 1}, {"supersedes_issue_id": "I000001"}, {"late_raised": False}):
            with self.assertRaises(ValidationError):
                Issue.model_validate({**raw, **addition})

    def test_generated_schema_has_closed_objects_and_no_scoring_fields(self):
        forbidden = {"weights", "final_score", "dimension", "procedural_state", "barrier_category", "confidence_score"}
        def inspect(node):
            if isinstance(node, dict):
                if node.get("type") == "object" and "properties" in node:
                    self.assertFalse(node["additionalProperties"])
                    self.assertFalse(forbidden.intersection(node["properties"]))
                for value in node.values():
                    inspect(value)
            elif isinstance(node, list):
                for value in node:
                    inspect(value)
        for model in [*MODEL_OUTPUT_SCHEMAS.values(), RunResult]:
            inspect(model.model_json_schema())
        # JSON input is subject to the same validation as Python input.
        self.assertEqual(PreAssessment.model_validate_json(json.dumps(pre())).issues, [])


class ConfigTests(unittest.TestCase):
    def test_defaults_are_stage_only_and_reasoning_off(self):
        with patch.dict(os.environ, {}, clear=True):
            config = ModelConfig()
        for stage in ("selection", "pre", "post", "editor"):
            self.assertEqual(config.get_temperature(stage), 0.0)
        self.assertEqual(config.get_temperature("debate"), 0.35)
        self.assertEqual(config.settings.reasoning_effort, "none")

    def test_missing_invalid_and_persona_override_configs_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            with self.assertRaises(FileNotFoundError):
                ModelConfig(path)
            for content in ('[]', '{"reasoning_effort":"high"}', '{"persona_overrides":{}}', '{"stages":{"debate":{"temperature":-1}}}'):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    ModelConfig(path)

    def test_environment_and_stage_precedence(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"PEER_REVIEW_MODEL": "environment-model"}):
            path = Path(folder) / "settings.json"
            path.write_text('{"default_model":"baseline","stages":{"post":{"model":"post-model"}}}')
            config = ModelConfig(path)
            self.assertEqual(config.get_model("pre"), "environment-model")
            self.assertEqual(config.get_model("pre", default_override="cli-model"), "cli-model")
            self.assertEqual(config.get_model("post", default_override="cli-model"), "post-model")
            with self.assertRaises(ValueError):
                config.get_model("unknown")

    def test_offline_imports_do_not_create_artifacts_or_require_credentials(self):
        code = "import importlib; importlib.import_module('exp-12'); from schemas import PreAssessment"
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run([sys.executable, "-c", code], cwd=folder, env=env, check=True)
            self.assertEqual(list(Path(folder).iterdir()), [])


class TokenTests(unittest.TestCase):
    def test_cache_and_reasoning_not_double_counted(self):
        tracker = TokenTracker(pricing={"fixture-model": {"input": 2.0, "output": 8.0,
                                "cache_read": 0.5, "cache_write": 3.0, "basis": "Synthetic test rates"}})
        tracker.record_attempt(attempt())
        self.assertEqual(tracker.get_total_tokens()["total"], 140)
        self.assertAlmostEqual(tracker.estimate_cost(), (70 * 2 + 40 * 8 + 20 * .5 + 10 * 3) / 1_000_000)

    def test_unknown_usage_and_prices_do_not_become_zero(self):
        tracker = TokenTracker()
        tracker.record_attempt(attempt())
        self.assertIsNone(tracker.estimate_cost())
        tracker.record_attempt(attempt(call_id="call-2", usage=None, status="TRANSPORT_ERROR", errors=["timeout"]))
        self.assertIsNone(tracker.get_total_tokens()["total"])
        self.assertEqual(tracker.get_summary()["unpriced_attempts"], 2)
        self.assertIn("unavailable", tracker.get_markdown_summary())

    def test_retry_cost_usage_and_raw_attempts_survive_save(self):
        tracker = TokenTracker()
        tracker.record_attempt(attempt(status="PARSE_ERROR", errors=["malformed JSON"]))
        tracker.record_attempt(attempt(attempt=2, retry_kind="VALIDATION", sent_temperature=0.3))
        summary = tracker.get_summary()
        self.assertEqual((summary["total_calls"], summary["total_attempts"], summary["retry_count"]), (1, 2, 1))
        self.assertEqual(summary["total_tokens"]["total"], 280)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "usage.json"
            tracker.save_to_file(path)
            raw = json.loads(path.read_text())
        self.assertEqual(raw["attempts"][0]["status"], "PARSE_ERROR")
        self.assertEqual(raw["attempts"][1]["sent_temperature"], 0.3)
        self.assertIsNone(raw["summary"]["total_cost"])
        with self.assertRaises(ValueError):
            tracker.record_attempt(attempt(attempt=2))

    def test_invalid_usage_subsets_rejected(self):
        raw = attempt().usage.model_dump()
        for update in ({"cache_read_tokens": 101}, {"reasoning_tokens": 41}, {"input_tokens": True}, {"output_tokens": -1}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                TokenUsage.model_validate({**raw, **update})

    def test_unknown_cache_rate_is_not_invented(self):
        tracker = TokenTracker(pricing={"fixture-model": {"input": 2.0, "output": 8.0, "basis": "Synthetic"}})
        tracker.record_attempt(attempt())
        self.assertIsNone(tracker.estimate_cost())



if __name__ == "__main__":
    unittest.main()
