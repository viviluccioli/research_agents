"""Behavioral tests of synchronous, append-only issue-centered review."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import importlib
architecture = importlib.import_module("exp-12")
GraphError, Ledger = architecture.GraphError, architecture.Ledger
from schemas import SelectionOutput
from test_checkpoint1 import candidate, core, pre, EVIDENCE
from config import ModelConfig

PANEL = ["Econometrician", "Theorist", "Historian"]


def record(persona, payload=None, status="OK", call=None):
    return {"persona": persona, "outcome": {
        "status": status, "payload": payload,
        "call_ids": [call or persona], "reason": None if status == "OK" else "Unavailable",
    }}


def debate(**changes):
    return {"arguments": [], "new_issues": [], "originator_updates": [], "issue_links": [], **changes}


def update(issue="I000001", stance="MAINTAIN", narrowed=None):
    return {"issue_id": issue, "stance": stance, "rationale": "Judgment after discussion.",
            "confidence": None if stance == "RETRACT" else 8.0, "revised_issue": narrowed}


def argument(issue="I000001", reply=None, action="DEFENSE"):
    return {"issue_id": issue, "responds_to_argument_id": reply, "action": action,
            "content": "The table supports this proposition.", "evidence": [EVIDENCE],
            "confidence": None if action in ("QUESTION", "CONCESSION") else 8.0}


def setup(rounds=2, issues=True, failed=None):
    ledger = Ledger(SelectionOutput(selected_personas=PANEL, selection_rationale="Private selection rationale"), rounds)
    records = []
    for index, persona in enumerate(PANEL):
        payload = pre()
        if index == 0 and issues:
            payload["issues"] = [candidate()]
        records.append(record(persona, None if persona == failed else payload,
                              "FAILED" if persona == failed else "OK", f"pre-{persona}"))
    ledger.commit_pre(records)
    return ledger


def round_records(ledger, number, outputs=None):
    return [record(p, (outputs or {}).get(p, debate()), call=f"r{number}-{p}") for p in ledger.eligible_personas]


def post_records(ledger, outputs=None):
    views = ledger.current_issues()
    issues = ledger.export()["issues"]
    records = []
    for persona in PANEL:
        if persona not in ledger.eligible_personas:
            skipped = record(persona, status="SKIPPED")
            skipped["outcome"]["call_ids"] = []
            records.append(skipped)
            continue
        own = {i["issue_id"] for i in issues if i["originator"] == persona}
        payload = {**core(), "originator_updates": [update(v["issue_id"]) for v in views if v["active"] and v["issue_id"] in own],
                   "late_observations": []}
        records.append(record(persona, (outputs or {}).get(persona, payload), call=f"post-{persona}"))
    return records


class LedgerTests(unittest.TestCase):
    def test_current_views_cannot_mutate_saved_evidence(self):
        ledger = setup()
        before = ledger.export()
        views = ledger.current_issues()
        views[0]["current"]["evidence"][0]["support"] = "Changed outside the ledger"
        views[0]["current"]["related_issue_ids"].append("I000099")
        self.assertEqual(ledger.export(), before)

    def test_editor_sees_failed_round_and_last_owner_reason_with_missing_post(self):
        ledger = setup(rounds=1)
        records = round_records(ledger, 1, {PANEL[0]: debate(originator_updates=[update()])})
        records[1] = record(PANEL[1], status="FAILED", call="r1-failed")
        ledger.commit_round(ledger.begin_round(), records)
        posts = post_records(ledger)
        posts[0] = record(PANEL[0], status="FAILED", call="post-failed")
        ledger.commit_post(ledger.begin_post(), posts)
        from schemas import AssessmentRecord, PostAssessment
        context = architecture.editor_context(ledger, [AssessmentRecord[PostAssessment].model_validate(p) for p in posts], {}, "Text")
        self.assertEqual(context["debate_availability"][0]["responses"][1]["status"], "FAILED")
        self.assertEqual(context["current_issue_views"][0]["originator_stance_rationale"], "Judgment after discussion.")
        self.assertIsNone(context["current_issue_views"][0]["post_confidence"])

    def test_editor_distinguishes_arguments_on_retracted_issues(self):
        ledger = setup(rounds=1)
        ledger.commit_round(ledger.begin_round(), round_records(ledger, 1, {
            PANEL[0]: debate(originator_updates=[update(stance="RETRACT")]),
            PANEL[1]: debate(arguments=[argument()]),
        }))
        posts = post_records(ledger)
        ledger.commit_post(ledger.begin_post(), posts)
        context = architecture.editor_context(ledger, [], {}, "Text")
        self.assertEqual(context["current_issue_views"], [])
        self.assertFalse(context["arguments"][0]["addresses_active_current_issue"])
        self.assertEqual(context["retracted_issues"][0]["rationale"], "Judgment after discussion.")

    def test_ids_are_deterministic_and_inputs_are_detached(self):
        def create(reverse):
            ledger = Ledger(SelectionOutput(selected_personas=PANEL, selection_rationale="Private"))
            records = [record(p, {**pre(), "issues": [candidate()]}, call=f"pre-{p}") for p in PANEL]
            ledger.commit_pre(list(reversed(records)) if reverse else records)
            records[0]["outcome"]["payload"]["issues"][0]["severity"] = 1
            return ledger.export()
        first, second = create(False), create(True)
        self.assertEqual(first, second)
        self.assertEqual([i["issue_id"] for i in first["issues"]], ["I000001", "I000002", "I000003"])
        self.assertEqual(first["issues"][0]["severity"], 7.2)

    def test_snapshot_is_frozen_and_does_not_expose_selection_rationale(self):
        ledger = setup()
        snapshot = ledger.begin_round()
        self.assertIs(snapshot, ledger.begin_round())
        self.assertNotIn("Private selection rationale", snapshot.structured_json)
        with self.assertRaises(FrozenInstanceError):
            snapshot.phase = "POST"
        copy = snapshot.as_dict()
        copy["issues"].clear()
        self.assertEqual(len(snapshot.as_dict()["issues"]), 1)
        self.assertIn(candidate()["plain_language_statement"], snapshot.structured_json)

    def test_same_round_references_rejected_and_batch_rolls_back(self):
        ledger = setup()
        snapshot = ledger.begin_round()
        before = ledger.export()
        outputs = {PANEL[0]: debate(new_issues=[candidate()]),
                   PANEL[1]: debate(arguments=[argument("I000002")])}
        with self.assertRaisesRegex(GraphError, "not visible"):
            ledger.commit_round(snapshot, round_records(ledger, 1, outputs))
        self.assertEqual(before, ledger.export())
        self.assertIs(snapshot, ledger.begin_round())
        # Corrected batch gets the same next ID: failed commits consume nothing.
        outputs[PANEL[1]] = debate()
        ledger.commit_round(snapshot, round_records(ledger, 1, outputs))
        self.assertEqual(ledger.export()["issues"][-1]["issue_id"], "I000002")

    def test_same_round_argument_reply_is_not_visible(self):
        ledger = setup()
        snapshot = ledger.begin_round()
        outputs = {PANEL[0]: debate(arguments=[argument()]),
                   PANEL[1]: debate(arguments=[argument(reply="A000001")])}
        with self.assertRaisesRegex(GraphError, "Argument A000001 is not visible"):
            ledger.commit_round(snapshot, round_records(ledger, 1, outputs))
        self.assertEqual(ledger.export()["arguments"], [])

    def test_full_history_carries_forward_without_array_truncation(self):
        ledger = setup(rounds=3)
        first = ledger.begin_round()
        ledger.commit_round(first, round_records(ledger, 1, {PANEL[1]: debate(arguments=[argument()] * 6)}))
        second = ledger.begin_round()
        self.assertEqual(len(second.as_dict()["arguments"]), 6)
        ledger.commit_round(second, round_records(ledger, 2, {PANEL[2]: debate(arguments=[argument(reply="A000001")])}))
        third = ledger.begin_round()
        self.assertEqual(len(third.as_dict()["arguments"]), 7)
        self.assertEqual(len(first.as_dict()["arguments"]), 0)
        self.assertNotIn("snapshot_json", third.structured_json)
        with self.assertRaisesRegex(GraphError, "stale"):
            ledger.validate_response(first, PANEL[0], debate())

    def test_cross_issue_replies_and_foreign_snapshots_rejected(self):
        ledger = setup()
        first = ledger.begin_round()
        ledger.commit_round(first, round_records(ledger, 1, {PANEL[0]: debate(new_issues=[candidate()], arguments=[argument()])}))
        second = ledger.begin_round()
        with self.assertRaisesRegex(GraphError, "same issue"):
            ledger.validate_response(second, PANEL[1], debate(arguments=[argument("I000002", "A000001")]))
        with self.assertRaisesRegex(GraphError, "foreign"):
            ledger.validate_response(setup().begin_round(), PANEL[1], debate())

    def test_only_owner_can_update_and_concession_is_not_retraction(self):
        ledger = setup()
        first = ledger.begin_round()
        with self.assertRaisesRegex(GraphError, "originator"):
            ledger.validate_response(first, PANEL[1], debate(originator_updates=[update(stance="RETRACT")]))
        ledger.commit_round(first, round_records(ledger, 1, {PANEL[1]: debate(arguments=[argument(action="CONCESSION")])}))
        self.assertTrue(ledger.current_issues()[0]["active"])
        self.assertIsNone(ledger.current_issues()[0]["originator_stance"])
        second = ledger.begin_round()
        ledger.commit_round(second, round_records(ledger, 2, {PANEL[0]: debate(originator_updates=[update(stance="RETRACT")])}))
        self.assertFalse(ledger.current_issues()[0]["active"])
        snapshot = ledger.begin_post()
        with self.assertRaisesRegex(GraphError, "retracted"):
            ledger.validate_response(snapshot, PANEL[0], {**core(), "originator_updates": [update()], "late_observations": []})

    def test_final_round_narrowing_keeps_original_and_does_not_inherit_testing(self):
        ledger = setup(rounds=1)
        original = deepcopy(ledger.export()["issues"][0])
        snap = ledger.begin_round()
        narrower = {**candidate(), "technical_statement": "Only subgroup inference is unsupported."}
        ledger.commit_round(snap, round_records(ledger, 1, {PANEL[0]: debate(originator_updates=[update(stance="NARROW", narrowed=narrower)])}))
        data = ledger.export()
        self.assertEqual(data["issues"][0], original)
        self.assertEqual(len(data["issues"]), 1)
        self.assertEqual(data["issue_revisions"][0]["issue_id"], "I000001")
        narrowed = ledger.current_issues()[0]
        self.assertTrue(narrowed["active"])
        self.assertEqual(narrowed["version"], 2)
        self.assertTrue(narrowed["late_raised"])
        self.assertEqual(narrowed["response_opportunity_count"], 0)
        self.assertEqual(len(data["opportunities"]), 2)  # Original version was exposed.
        ledger.commit_post(ledger.begin_post(), post_records(ledger))
        self.assertTrue(ledger.current_issues()[0]["late_raised"])
        self.assertEqual(ledger.current_issues()[0]["originator_final_stance"], "MAINTAIN")

    def test_historical_reply_keeps_its_version_after_narrowing(self):
        ledger = setup(rounds=2)
        first = ledger.begin_round()
        ledger.commit_round(first, round_records(ledger, 1, {
            PANEL[0]: debate(originator_updates=[update(stance="NARROW", narrowed={**candidate(), "severity": 4.0})]),
            PANEL[1]: debate(arguments=[argument()]),
        }))
        ledger.commit_round(ledger.begin_round(), round_records(ledger, 2, {
            PANEL[1]: debate(arguments=[argument(reply="A000001")]),
            PANEL[2]: debate(arguments=[argument()]),
        }))
        self.assertEqual([a["issue_version"] for a in ledger.export()["arguments"]], [1, 1, 2])
        self.assertEqual(ledger.current_issues()[0]["current"]["severity"], 4.0)
        self.assertEqual(ledger.current_issues()[0]["engagement_count"], 1)

    def test_changed_substance_can_be_revised_without_narrow_stance(self):
        ledger = setup(rounds=1)
        revised = {**candidate(), "technical_statement": "The concern now concerns the mechanism.", "severity": 5.0}
        ledger.commit_round(ledger.begin_round(), round_records(ledger, 1, {
            PANEL[0]: debate(originator_updates=[update(stance="UNCERTAIN", narrowed=revised)])
        }))
        self.assertEqual(len(ledger.export()["issues"]), 1)
        self.assertEqual(ledger.current_issues()[0]["version"], 2)
        self.assertTrue(ledger.current_issues()[0]["active"])

    def test_post_requires_complete_owner_coverage_but_missing_post_stays_missing(self):
        ledger = setup(rounds=0)
        snap = ledger.begin_post()
        with self.assertRaisesRegex(GraphError, "every own active issue"):
            ledger.validate_response(snap, PANEL[0], {**core(), "originator_updates": [], "late_observations": []})
        records = post_records(ledger)
        records[0] = record(PANEL[0], status="FAILED", call="post-failed")
        ledger.commit_post(snap, records)
        view = ledger.current_issues()[0]
        self.assertIsNone(view["originator_final_stance"])
        self.assertIsNone(view["post_confidence"])
        self.assertEqual(view["current_confidence"], 8.0)
        self.assertFalse(view["late_raised"])
        self.assertEqual(view["response_opportunity_count"], 0)

    def test_post_narrowing_and_observations_remain_late(self):
        ledger = setup(rounds=0)
        records = post_records(ledger, {PANEL[0]: {**core(), "originator_updates": [update(stance="NARROW", narrowed=candidate())], "late_observations": [candidate()]}})
        ledger.commit_post(ledger.begin_post(), records)
        views = ledger.current_issues()
        self.assertEqual(len(views), 2)
        self.assertTrue(views[0]["active"])
        self.assertTrue(all(v["late_raised"] for v in views))
        self.assertIsNone(views[1]["originator_final_stance"])
        self.assertEqual(views[0]["originator_final_stance"], "NARROW")
        self.assertEqual(views[0]["post_confidence"], 8.0)
        self.assertEqual(views[0]["version"], 2)

    def test_final_round_new_issue_remains_late_after_post_maintains_it(self):
        ledger = setup(rounds=1, issues=False)
        ledger.commit_round(ledger.begin_round(), round_records(ledger, 1, {
            PANEL[0]: debate(new_issues=[candidate()])
        }))
        ledger.commit_post(ledger.begin_post(), post_records(ledger))
        view = ledger.current_issues()[0]
        self.assertTrue(view["late_raised"])
        self.assertTrue(view["active"])
        self.assertEqual(view["originator_final_stance"], "MAINTAIN")
        self.assertEqual(view["response_opportunity_count"], 0)

    def test_failure_empty_response_and_skipped_call_are_distinct(self):
        ledger = setup(rounds=1)
        records = round_records(ledger, 1)
        records[1] = record(PANEL[1], status="FAILED", call="r1-failed")
        ledger.commit_round(ledger.begin_round(), records)
        view = ledger.current_issues()[0]
        self.assertEqual((view["response_opportunity_count"], view["failed_response_count"], view["engagement_count"]), (1, 1, 0))
        self.assertTrue(view["active"])

    def test_possible_duplicate_links_preserve_independent_issues_and_attribution(self):
        ledger = setup()
        snap = ledger.begin_round()
        ledger.commit_round(snap, round_records(ledger, 1, {PANEL[1]: debate(new_issues=[candidate()])}))
        snap = ledger.begin_round()
        link = {"issue_ids": ["I000001", "I000002"], "relationship": "POSSIBLE_DUPLICATE", "rationale": "Same table and objection."}
        ledger.commit_round(snap, round_records(ledger, 2, {PANEL[2]: debate(issue_links=[link])}))
        self.assertEqual(len(ledger.export()["issues"]), 2)
        self.assertEqual(ledger.export()["issue_links"][0]["proposer"], PANEL[2])
        self.assertEqual(ledger.current_issues()[0]["related_issue_ids"], ["I000002"])
        self.assertEqual(ledger.current_issues()[1]["related_issue_ids"], ["I000001"])

    def test_failed_pre_excluded_and_post_explicitly_skipped(self):
        ledger = setup(rounds=1, failed=PANEL[2])
        self.assertEqual(ledger.eligible_personas, tuple(PANEL[:2]))
        ledger.commit_round(ledger.begin_round(), round_records(ledger, 1))
        ledger.commit_post(ledger.begin_post(), post_records(ledger))
        self.assertEqual(ledger.export()["post_assessments"][2]["outcome"]["status"], "SKIPPED")

    def test_no_issues_valid_and_event_order_monotonic(self):
        ledger = setup(rounds=1, issues=False)
        ledger.commit_round(ledger.begin_round(), round_records(ledger, 1, {PANEL[0]: debate(new_issues=[candidate()])}))
        ledger.commit_post(ledger.begin_post(), post_records(ledger))
        data = ledger.export()
        events = [e["event_sequence"] for key in ("issues", "issue_revisions", "arguments", "issue_updates", "issue_links") for e in data[key]]
        self.assertEqual(sorted(events), list(range(1, len(events) + 1)))
        data["issues"].clear()
        self.assertEqual(len(ledger.export()["issues"]), 1)
        with self.assertRaises(GraphError):
            ledger.begin_post()

    def test_stage_order_roster_and_editor_ids(self):
        ledger = setup(rounds=1)
        with self.assertRaises(GraphError):
            ledger.begin_post()
        snapshot = ledger.begin_round()
        with self.assertRaisesRegex(GraphError, "expected panel"):
            ledger.commit_round(snapshot, round_records(ledger, 1)[:2])
        ledger.commit_round(snapshot, round_records(ledger, 1))
        ledger.commit_post(ledger.begin_post(), post_records(ledger))
        editor = {"integrated_rationale": "Review complete.", "verdict": "RESUBMIT", "author_letter": "Please revise.",
                  "revision_path": "Revise interpretation.", "constructive_summary": "Useful design.", "decisive_issue_ids": ["I000001"]}
        ledger.validate_editor(editor)
        with self.assertRaisesRegex(GraphError, "unknown issue"):
            ledger.validate_editor({**editor, "decisive_issue_ids": ["I000099"]})


    def test_reused_call_provenance_is_rejected_without_changes(self):
        ledger = setup()
        snapshot = ledger.begin_round()
        records = round_records(ledger, 1)
        records[0]["outcome"]["call_ids"] = [f"pre-{PANEL[0]}"]
        before = ledger.export()
        with self.assertRaisesRegex(GraphError, "reused"):
            ledger.commit_round(snapshot, records)
        self.assertEqual(before, ledger.export())

    def test_all_pre_failures_and_skipped_debate_do_not_create_opportunities(self):
        empty = Ledger(SelectionOutput(selected_personas=PANEL, selection_rationale="Private"))
        empty.commit_pre([record(p, status="FAILED", call=f"pre-{p}") for p in PANEL])
        with self.assertRaisesRegex(GraphError, "valid independent PRE"):
            empty.begin_round()
        ledger = setup(rounds=1)
        records = round_records(ledger, 1)
        records[1] = record(PANEL[1], status="SKIPPED")
        records[1]["outcome"]["call_ids"] = []
        ledger.commit_round(ledger.begin_round(), records)
        view = ledger.current_issues()[0]
        self.assertEqual(view["response_opportunity_count"], 1)
        self.assertEqual(view["failed_response_count"], 0)


class FakeClient:
    """Exercises the real prompts, validator, ledger and persistence without HTTP."""
    def __init__(self, failures=None, with_issues=True):
        self.requests = []
        self.failures = failures or {}
        self.with_issues = with_issues
        self.counts = {}

    def complete(self, request):
        self.requests.append(deepcopy(request))
        system = request["messages"][0]["content"]
        schema = json.loads(system.split("\n")[-1])
        stage = {"SelectionOutput": "SELECTION", "PreAssessment": "PRE", "DebateOutput": "DEBATE",
                 "PostAssessment": "POST", "EditorOutput": "EDITOR"}[schema["title"]]
        match = re.search(r"YOUR ROLE: (\w+)", system)
        persona = match.group(1) if match else None
        key = (stage, persona)
        self.counts[key] = self.counts.get(key, 0) + 1
        content = request["messages"][1]["content"].split("\nYour previous response failed validation.")[0]
        context = json.loads(content)
        failure = self.failures.get((stage, persona, self.counts[key]), self.failures.get(key))
        if failure == "transport":
            raise architecture.TransportError("Synthetic timeout")
        if stage == "SELECTION":
            payload = {"selected_personas": PANEL, "selection_rationale": "PRIVATE_SELECTION_RATIONALE"}
        elif stage == "PRE":
            payload = pre()
            payload["structural_strength"] = "PRIVATE_PRE_" + persona
            if self.with_issues and persona == PANEL[0]:
                payload["issues"] = [candidate()]
        elif stage == "DEBATE":
            payload = debate()
            if self.with_issues:
                ledger = context["ledger"]
                round_number = len(ledger["round_outcomes"]) + 1
                if persona == PANEL[0] and round_number == 1:
                    revised = {**candidate(), "severity": 4.5,
                               "technical_statement": "Only the mechanism claim remains unsupported."}
                    payload["originator_updates"] = [update(stance="NARROW", narrowed=revised)]
                elif persona == PANEL[1]:
                    payload["arguments"] = [argument()]
        elif stage == "POST":
            views = context["ledger"]["issue_views"]
            payload = {**core(), "assessment_rationale": "PRIVATE_POST_" + persona,
                       "originator_updates": [update(v["issue_id"]) for v in views
                                              if v["active"] and v["current"]["originator"] == persona],
                       "late_observations": []}
        else:
            payload = {"integrated_rationale": "The current mechanism concern needs a bounded repair.",
                       "verdict": "RESUBMIT", "author_letter": "Please clarify the mechanism.",
                       "revision_path": "Test or narrow the mechanism.", "constructive_summary": "Useful estimates.",
                       "decisive_issue_ids": ["I000001"] if context["current_issue_views"] else []}
        if failure == "schema":
            payload = {"wrong_field": True}
        elif failure == "graph":
            payload = debate(arguments=[argument("I999999")])
        text = "not json" if failure == "parse" else "```json\n" + json.dumps(payload) + "\n```"
        return {"model": "fixture-model", "choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30,
                          "prompt_tokens_details": {"cached_tokens": 10, "cache_creation_tokens": 0},
                          "completion_tokens_details": {"reasoning_tokens": 0}}}


class WorkflowTests(unittest.TestCase):
    def test_malformed_usage_details_do_not_discard_valid_totals(self):
        usage = architecture.normalize_usage({"usage": {
            "prompt_tokens": 100, "completion_tokens": 30,
            "prompt_tokens_details": ["unexpected"], "completion_tokens_details": "unexpected",
        }})
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 30)
        self.assertIsNone(usage.reasoning_tokens)
        self.assertIsNone(usage.cache_read_tokens)

    def test_abstract_stops_at_heading_at_end_of_file(self):
        abstract = architecture.extract_abstract("Title\nAbstract\nOnly the abstract.\n1 Introduction")
        self.assertEqual(abstract["text"].strip(), "Only the abstract.")

    def run_case(self, client, *, manuscript=None, embedding_backend=None, **overrides):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        config = ModelConfig(overrides={"default_model": "fixture-model", **overrides})
        # Any accidental use of the live adapter makes an offline test fail.
        with patch.object(architecture.ChatClient, "complete", side_effect=AssertionError("Unexpected HTTP")):
            result = architecture.run_review(manuscript or "Abstract\nSynthetic manuscript with subgroup estimates.\n1 Introduction\nEvidence.",
                                             client=client, config=config, output_dir=self.folder.name,
                                             embedding_backend=embedding_backend)
        self.directory = Path(self.folder.name) / result.run_id
        saved = json.loads((self.directory / "result.json").read_text())
        self.assertEqual(saved, result.model_dump(mode="json"))
        return result

    def test_full_workflow_independence_versions_and_provenance(self):
        client = FakeClient()
        result = self.run_case(client)
        self.assertEqual(result.status, "COMPLETE")
        self.assertEqual(len(result.calls), 14)
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(len(result.issue_revisions), 1)
        self.assertEqual(result.issues[0].severity, 7.2)
        self.assertEqual(result.issue_revisions[0].severity, 4.5)
        self.assertEqual(result.issue_revisions[0].issue_id, result.issues[0].issue_id)
        self.assertEqual([a.issue_version for a in result.arguments], [1, 2])
        pre_calls = client.requests[1:4]
        self.assertEqual(len({r["messages"][1]["content"] for r in pre_calls}), 1)
        for request in pre_calls:
            self.assertNotIn("PRIVATE_SELECTION", json.dumps(request))
            self.assertNotIn("PRIVATE_PRE", json.dumps(request))
            self.assertNotIn("Synthetic manuscript", request["messages"][0]["content"])
        for start in (4, 7):
            contexts = [r["messages"][1]["content"] for r in client.requests[start:start + 3]]
            self.assertEqual(len(set(contexts)), 1)
        second = json.loads(client.requests[7]["messages"][1]["content"])["ledger"]
        self.assertEqual(second["arguments"][0]["issue_version"], 1)
        self.assertEqual(second["issue_views"][0]["version"], 2)
        self.assertEqual(second["issue_views"][0]["response_opportunity_count"], 0)
        self.assertIn(candidate()["plain_language_statement"], json.dumps(second))
        for request in client.requests[10:13]:
            self.assertNotIn("PRIVATE_POST", request["messages"][1]["content"])
        editor = json.loads(client.requests[-1]["messages"][1]["content"])
        self.assertNotIn("issues", editor)
        self.assertEqual(editor["current_issue_views"][0]["current"]["severity"], 4.5)
        self.assertEqual([a["addresses_current_version"] for a in editor["arguments"]], [False, True])
        for call in result.calls:
            self.assertTrue((self.directory / call.raw_response_path).exists())
            request = json.loads((self.directory / call.request_context.path).read_text())
            self.assertEqual(request["reasoning_effort"], "none")
            self.assertEqual(request["temperature"], .35 if call.stage == "DEBATE" else 0)
        self.assertTrue((self.directory / "report.md").exists())
        forbidden = {"final_score", "weights", "adjusted_weights", "barrier_category", "confidence_score"}
        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for child in value.values(): inspect(child)
            elif isinstance(value, list):
                for child in value: inspect(child)
        inspect(result.model_dump())

    def test_empty_success_and_failed_pre_are_distinct(self):
        result = self.run_case(FakeClient(with_issues=False))
        self.assertEqual(result.status, "COMPLETE")
        self.assertEqual(result.issues, [])
        self.assertTrue(all(r.outcome.payload is not None for r in result.pre_assessments))
        result = self.run_case(FakeClient(failures={("PRE", PANEL[0]): "parse"}, with_issues=False))
        self.assertEqual(result.status, "PARTIAL")
        self.assertIsNone(result.pre_assessments[0].outcome.payload)
        self.assertEqual(result.post_assessments[0].outcome.status, "SKIPPED")
        self.assertEqual(len(result.rounds[0].responses), 2)

    def test_retries_preserve_raw_usage_and_graph_errors(self):
        client = FakeClient(failures={("PRE", PANEL[0], 1): "parse",
                                      ("DEBATE", PANEL[1], 1): "graph",
                                      ("EDITOR", None, 1): "transport"})
        result = self.run_case(client)
        self.assertEqual(result.status, "COMPLETE")
        self.assertEqual(len(result.calls), 17)
        failures = [c for c in result.calls if c.status != "OK"]
        self.assertEqual([c.status for c in failures], ["PARSE_ERROR", "GRAPH_ERROR", "TRANSPORT_ERROR"])
        self.assertIsNotNone(failures[0].usage)
        self.assertIsNone(failures[2].usage)
        retried = [c for c in result.calls if c.attempt == 2]
        self.assertEqual(retried[0].sent_temperature, 0)
        self.assertEqual(retried[1].sent_temperature, .35)
        self.assertEqual(retried[2].sent_temperature, 0)
        self.assertTrue(all(c.sent_temperature == c.requested_temperature for c in result.calls))
        self.assertTrue((self.directory / failures[0].raw_response_path).exists())
        tokens = json.loads((self.directory / "tokens.json").read_text())
        self.assertEqual(tokens["summary"]["retry_count"], 3)
        self.assertIsNone(tokens["summary"]["total_tokens"]["total"])

    def test_all_failed_pre_skips_downstream_without_invented_ratings(self):
        result = self.run_case(FakeClient(failures={("PRE", p): "schema" for p in PANEL}))
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.rounds, [])
        self.assertIsNone(result.editor.payload)
        self.assertTrue(all(p.outcome.payload is None for p in result.post_assessments))
        self.assertFalse(any(c.stage in ("POST", "EDITOR", "DEBATE") for c in result.calls))

    def test_failed_post_and_editor_preserve_history_without_fabrication(self):
        result = self.run_case(FakeClient(failures={("POST", PANEL[0]): "parse", ("EDITOR", None): "schema"}))
        self.assertEqual(result.status, "PARTIAL")
        self.assertIsNone(result.post_assessments[0].outcome.payload)
        self.assertIsNone(result.editor.payload)
        self.assertTrue(result.issue_revisions)
        self.assertFalse(any(u.phase == "POST" and u.persona == PANEL[0] for u in result.issue_updates))
        self.assertIn("Editor unavailable", (self.directory / "report.md").read_text())

    def test_editor_sees_failed_debate_even_when_no_issues_exist(self):
        client = FakeClient(failures={("DEBATE", p): "parse" for p in PANEL}, with_issues=False)
        result = self.run_case(client)
        self.assertEqual(result.status, "PARTIAL")
        self.assertEqual(result.issues, [])
        context = json.loads(client.requests[-1]["messages"][1]["content"])
        self.assertEqual(len(context["debate_availability"]), 2)
        self.assertTrue(all(r["status"] == "FAILED" for round_ in context["debate_availability"] for r in round_["responses"]))

    def test_retracted_concern_is_historical_in_editor_context_and_report(self):
        class RetractionClient(FakeClient):
            def complete(self, request):
                raw = super().complete(request)
                payload = architecture.parse_json_object(raw["choices"][0]["message"]["content"])
                for item in payload.get("originator_updates", []):
                    if item["stance"] == "NARROW":
                        item.update(stance="RETRACT", confidence=None, revised_issue=None)
                raw["choices"][0]["message"]["content"] = json.dumps(payload)
                return raw
        client = RetractionClient()
        result = self.run_case(client)
        self.assertEqual(result.status, "COMPLETE")
        context = json.loads(client.requests[-1]["messages"][1]["content"])
        self.assertEqual(context["current_issue_views"], [])
        self.assertFalse(any(a["addresses_active_current_issue"] for a in context["arguments"]))
        report = (self.directory / "report.md").read_text()
        current_section = report.split("## Current issue record")[1].split("## Retracted issues")[0]
        self.assertNotIn(candidate()["technical_statement"], current_section)
        self.assertIn("## Retracted issues (history)", report)

    def test_context_limit_and_selection_fallback_are_explicit(self):
        client = FakeClient()
        result = self.run_case(client, max_context_characters=1)
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.selection.status, "FALLBACK")
        self.assertEqual(client.requests, [])
        self.assertTrue(any("Context limit" in d for d in result.diagnostic_messages))

    def test_zero_rounds_and_abstract_extraction(self):
        result = self.run_case(FakeClient(), debate_rounds=0)
        self.assertEqual(result.rounds, [])
        self.assertEqual(len(result.calls), 8)
        self.assertTrue(all(c.sent_temperature == 0 for c in result.calls))
        text = "Title\nAbstract\nAn actual abstract.\n1 Introduction\nBody"
        abstract = architecture.extract_abstract(text)
        self.assertEqual(abstract["text"].strip(), "An actual abstract.")
        fallback = architecture.extract_abstract("no heading" * 1000)
        self.assertEqual(fallback["method"], "opening_text_fallback")
        self.assertTrue(fallback["truncated"])
        self.assertEqual(len(fallback["text"]), 6000)

    def test_all_persona_examples_validate(self):
        from schemas import PreAssessment
        for persona in architecture.ROLE_PROFILES:
            examples = architecture.role_examples(persona)
            self.assertEqual(len(examples), 3)
            for example in examples:
                PreAssessment.model_validate(example["response"])
            self.assertEqual(examples[0]["response"]["issues"], [])

    def test_repeated_schema_retries_keep_temperature_above_one(self):
        result = self.run_case(FakeClient(failures={("PRE", PANEL[0]): "schema"}),
                               stages={"pre": {"temperature": 1.4}})
        attempts = [c for c in result.calls if c.stage == "PRE" and c.persona == PANEL[0]]
        self.assertEqual(len(attempts), 3)
        self.assertEqual([c.sent_temperature for c in attempts], [1.4, 1.4, 1.4])
        self.assertEqual([c.retry_kind for c in attempts], ["INITIAL", "VALIDATION", "VALIDATION"])
        self.assertTrue(all(c.errors for c in attempts))

    def test_novelty_domains_belong_to_cases_and_can_span_domains(self):
        self.assertFalse(hasattr(architecture, "EXAMPLE_DOMAIN"))
        examples = architecture.role_examples("Historian")
        middle = examples[1]["response"]
        self.assertEqual(middle["novelty_by_domain"]["methodological"]["level"], "NONE")
        self.assertEqual(middle["novelty_by_domain"]["empirical"]["level"], "MEDIUM")
        self.assertEqual(middle["issues"][0]["novelty_domains"], ["methodological"])
        self.assertEqual(examples[2]["response"]["issues"][0]["novelty_domains"], ["methodological", "empirical"])
        # Reusing a teaching case under another persona never changes its domains.
        with patch.dict(architecture.ROLE_EXAMPLE_CASES, {"Theorist": architecture.ROLE_EXAMPLE_CASES["Historian"]}):
            reused = architecture.role_examples("Theorist")
        for original, copied in zip(examples, reused):
            self.assertEqual(original["response"]["novelty_by_domain"], copied["response"]["novelty_by_domain"])

    def test_prompt_hash_covers_every_persona_stage_and_is_order_independent(self):
        def fingerprint():
            return architecture.digest(architecture.compact_json(architecture.prompt_configuration()))
        baseline = fingerprint()
        actual = architecture.stage_system
        for stage in ("PRE", "DEBATE", "POST"):
            for role in architecture.ROLE_PROFILES:
                def changed(s, persona=None):
                    return actual(s, persona) + ("\nChanged persona instruction." if (s, persona) == (stage, role) else "")
                with self.subTest(stage=stage, role=role), patch.object(architecture, "stage_system", side_effect=changed):
                    self.assertNotEqual(baseline, fingerprint())
        with patch.object(architecture, "ROLE_PROFILES", dict(reversed(list(architecture.ROLE_PROFILES.items())))), \
             patch.object(architecture, "MODEL_OUTPUT_SCHEMAS", dict(reversed(list(architecture.MODEL_OUTPUT_SCHEMAS.items())))):
            self.assertEqual(baseline, fingerprint())
        self.assertEqual(baseline, fingerprint())
        examples = architecture.role_examples
        def changed_example(role):
            values = examples(role)
            if role == "Historian":
                values[0]["manuscript_vignette"] += " Changed example."
            return values
        with patch.object(architecture, "role_examples", side_effect=changed_example):
            self.assertNotEqual(baseline, fingerprint())
        schema = architecture.output_json_schema
        with patch.object(architecture, "output_json_schema", side_effect=lambda stage: {**schema(stage), "description": "changed"}):
            self.assertNotEqual(baseline, fingerprint())

    def test_saved_prompt_configuration_matches_sent_systems(self):
        result = self.run_case(FakeClient())
        prompts = json.loads((self.directory / "prompt-configuration.json").read_text())
        self.assertEqual(result.prompt_sha256, architecture.digest(architecture.compact_json(prompts)))
        for call in result.calls:
            request = json.loads((self.directory / call.request_context.path).read_text())
            expected = prompts["stages"][call.stage]
            if call.persona:
                expected = expected[call.persona]
            self.assertEqual(request["messages"][0]["content"], expected)

    def test_abstract_only_keeps_original_post_context(self):
        client = FakeClient()
        result = self.run_case(client, post_manuscript_context="abstract_only")
        snapshot = json.loads((self.directory / "post-snapshot.json").read_text())
        for call in [c for c in result.calls if c.stage == "POST"]:
            request = json.loads((self.directory / call.request_context.path).read_text())
            context = json.loads(request["messages"][1]["content"])
            self.assertEqual(set(context), {"ledger", "own_pre", "untrusted_abstract"})
            self.assertEqual(context["ledger"], snapshot)
            own_pre = next(p.outcome.payload for p in result.pre_assessments if p.persona == call.persona)
            self.assertEqual(context["own_pre"], own_pre.model_dump(mode="json"))

    def test_retrieved_post_provenance_matches_sent_passages_and_budget(self):
        manuscript = "Abstract\nSynthetic manuscript.\n1 Introduction\n\n" + "\n\n".join(
            f"Observation {i}. Subgroup estimates and the income gap support the mechanism claim. " * 10
            for i in range(12))
        result = self.run_case(FakeClient(), manuscript=manuscript, retrieval_budget_characters=2500)
        self.assertEqual(len(result.post_context_artifacts), 3)
        snapshot = json.loads((self.directory / "post-snapshot.json").read_text())
        for call, artifact in zip([c for c in result.calls if c.stage == "POST"], result.post_context_artifacts):
            request = json.loads((self.directory / call.request_context.path).read_text())
            raw_context = request["messages"][1]["content"]
            context = json.loads(raw_context)
            trace = json.loads((self.directory / artifact.path).read_text())
            self.assertNotIn("PRIVATE_POST", raw_context)
            self.assertNotIn("PRIVATE_POST", json.dumps(trace))
            self.assertEqual(context["ledger"], snapshot)
            self.assertEqual(trace["post_context_sha256"], architecture.digest(raw_context))
            self.assertEqual(trace["post_context_characters"], len(raw_context))
            chunks = context["untrusted_manuscript_evidence"]
            self.assertTrue(chunks)
            self.assertEqual(chunks, trace["selected_chunks"])
            self.assertLessEqual(len(architecture.compact_json(chunks)), 2500)
            self.assertLessEqual(sum(len(c["text"]) for c in chunks), len(manuscript) // 4)
            for chunk in chunks:
                self.assertEqual(chunk["text"], manuscript[chunk["start"]:chunk["end"]])

    def test_retrieval_failure_preserves_null_post_and_explicit_trace(self):
        with patch.object(architecture.RetrievalIndex, "retrieve", side_effect=architecture.RetrievalError("Synthetic failure")):
            result = self.run_case(FakeClient())
        self.assertEqual(result.status, "PARTIAL")
        self.assertFalse(any(c.stage == "POST" for c in result.calls))
        for record, artifact in zip(result.post_assessments, result.post_context_artifacts):
            self.assertEqual(record.outcome.status, "FAILED")
            self.assertIsNone(record.outcome.payload)
            trace = json.loads((self.directory / artifact.path).read_text())
            self.assertEqual(trace["status"], "FAILED")
            self.assertEqual(trace["selected_chunks"], [])
            self.assertIn("Synthetic failure", trace["error"])

    def test_missing_embedding_backend_never_silently_changes_method(self):
        result = self.run_case(FakeClient(), retrieval_method="hybrid")
        self.assertEqual(result.status, "PARTIAL")
        for artifact in result.post_context_artifacts:
            trace = json.loads((self.directory / artifact.path).read_text())
            self.assertEqual(trace["status"], "FAILED")
            self.assertEqual(trace["retrieval_method"], "hybrid")
            self.assertEqual(trace["selected_chunks"], [])

    def test_full_manuscript_ablation_retains_bound_and_settings_hash(self):
        result = self.run_case(FakeClient(), post_manuscript_context="full_manuscript", manuscript_characters=20)
        for call in [c for c in result.calls if c.stage == "POST"]:
            request = json.loads((self.directory / call.request_context.path).read_text())
            context = json.loads(request["messages"][1]["content"])
            self.assertEqual(len(context["untrusted_manuscript"]), 20)
            self.assertTrue(call.request_context.truncated)
        metadata = json.loads((self.directory / "run-metadata.json").read_text())
        self.assertEqual(result.settings_sha256, architecture.digest(architecture.compact_json(metadata)))
        changed = deepcopy(metadata)
        changed["settings"]["post_manuscript_context"] = "abstract_only"
        self.assertNotEqual(result.settings_sha256, architecture.digest(architecture.compact_json(changed)))


if __name__ == "__main__":
    unittest.main()
