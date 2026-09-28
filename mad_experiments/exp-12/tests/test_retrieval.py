"""Offline retrieval checks: raw source provenance, budgets, and POST independence."""
from copy import deepcopy
import json
import math
import unittest

from retrieval import RetrievalError, RetrievalIndex, build_queries, chunk_manuscript


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def manuscript():
    paragraphs = [
        "Loan contracts constrain household borrowing and change credit access in the village.",
        "Mortgages finance residential purchases while savings accounts buffer unexpected costs.",
        "Heliotrope seedlings thrive under violet lamps inside the northern greenhouse chamber.",
        "Astronomers record lunar craters through telescopes during the winter observation period.",
        "Railway timetables allocate passenger journeys between stations across the coastal route.",
        "Pottery fragments reveal ceramic techniques among settlements beside the mountain river.",
        "Musicians compare string harmonies using chamber recordings captured during rehearsals.",
        "Wildlife researchers identify nesting patterns among seabirds around the offshore island.",
        "Athletes evaluate marathon pacing through training records from several summer seasons.",
        "Architects describe courtyard ventilation using airflow measurements within brick houses.",
        "Linguists annotate vowel shifts across spoken dialects collected during field interviews.",
        "Sailors navigate ocean currents with instrument readings from vessels crossing the bay.",
    ]
    return "\n\n".join(paragraphs)


class SemanticFixture:
    fingerprint = "offline-two-topic-fixture-v1"

    def embed(self, texts):
        # Loan/mortgage are semantic neighbors even when the exact query word differs.
        return [[1.0, 0.0] if any(term in text.casefold() for term in ("loan", "mortgage", "credit"))
                else [0.0, 1.0] for text in texts]


class ChunkingTests(unittest.TestCase):
    def test_chunks_cover_original_source_with_stable_ids_offsets_and_neighbors(self):
        text = manuscript() + "\n\nUnicode evidence: café, λ, and an escaped quote \" remain verbatim."
        chunks = chunk_manuscript(text, chunk_characters=130)
        self.assertEqual(chunks, chunk_manuscript(text, chunk_characters=130))
        self.assertEqual("".join(chunk["text"] for chunk in chunks), text)
        self.assertEqual(chunks[0]["start"], 0)
        self.assertEqual(chunks[-1]["end"], len(text))
        for index, chunk in enumerate(chunks):
            with self.subTest(chunk=chunk["chunk_id"]):
                self.assertEqual(chunk["chunk_id"], f"C{index + 1:06d}")
                self.assertEqual(chunk["text"], text[chunk["start"]:chunk["end"]])
                self.assertEqual(chunk["characters"], chunk["end"] - chunk["start"])
                self.assertGreater(chunk["characters"], 0)
                self.assertLessEqual(chunk["characters"], 130)
                self.assertEqual(chunk["previous_chunk_id"], chunks[index - 1]["chunk_id"] if index else None)
                self.assertEqual(chunk["next_chunk_id"], chunks[index + 1]["chunk_id"] if index + 1 < len(chunks) else None)
                if index:
                    self.assertEqual(chunk["start"], chunks[index - 1]["end"])

    def test_boundaries_do_not_depend_on_semantic_section_names(self):
        text = "Methods\n" + manuscript() + "\n\nResults\n" + manuscript()
        renamed = text.replace("Methods", "Lantern").replace("Results", "Sunsets")
        original = chunk_manuscript(text, chunk_characters=140)
        alternative = chunk_manuscript(renamed, chunk_characters=140)
        self.assertEqual([(c["start"], c["end"]) for c in original],
                         [(c["start"], c["end"]) for c in alternative])

    def test_empty_and_unbroken_sources_are_covered_without_invented_text(self):
        self.assertEqual(chunk_manuscript(""), [])
        for text in ("x", "x" * 1000, " \n\t " * 100):
            with self.subTest(length=len(text)):
                chunks = chunk_manuscript(text, chunk_characters=80)
                self.assertEqual("".join(c["text"] for c in chunks), text)
                self.assertTrue(all(c["start"] < c["end"] for c in chunks))


class RetrievalTests(unittest.TestCase):
    def test_lexical_retrieval_finds_evidence_and_preserves_full_provenance(self):
        text = manuscript()
        index = RetrievalIndex(text, chunk_characters=180)
        queries = [{"source": "issue:I000001:v2", "text": "heliotrope seedlings"}]
        trace = index.retrieve(queries)
        self.assertEqual(trace["status"], "OK")
        self.assertEqual(trace["queries"], queries)
        self.assertIn("Heliotrope", trace["selected_chunks"][0]["text"])
        self.assertEqual(trace["context_characters"], len(compact(trace["selected_chunks"])))
        self.assertEqual(trace["evidence_characters"], sum(c["characters"] for c in trace["selected_chunks"]))
        original = {c["chunk_id"]: c for c in index.chunks}
        for rank, chunk in enumerate(trace["selected_chunks"], 1):
            self.assertEqual(chunk["selection_rank"], rank)
            self.assertGreater(chunk["fused_score"], 0)
            self.assertGreaterEqual(chunk["retrieval_rank"], 1)
            self.assertEqual(set(chunk["method_scores"]), {"lexical"})
            self.assertEqual(chunk["text"], text[chunk["start"]:chunk["end"]])
            for field, value in original[chunk["chunk_id"]].items():
                self.assertEqual(chunk[field], value)
        self.assertEqual(trace, index.retrieve(queries))

    def test_fixed_evidence_budget_counts_serialized_provenance_and_unicode(self):
        text = manuscript() + "\n\n" + "Café café credit λ household evidence. " * 10
        index = RetrievalIndex(text, chunk_characters=160)
        queries = [{"source": "own_pre:repair_scope", "text": "credit café household evidence"}]
        for budget in (2, 120, 600, 1100, 12000):
            with self.subTest(budget=budget):
                trace = index.retrieve(queries, budget_characters=budget, max_manuscript_fraction=0.25)
                self.assertLessEqual(len(compact(trace["selected_chunks"])), budget)
                self.assertEqual(trace["context_characters"], len(compact(trace["selected_chunks"])))
                self.assertLessEqual(trace["evidence_characters"], math.floor(len(text) * 0.25))
                self.assertLess(trace["evidence_characters"], len(text))
                self.assertEqual(trace["status"], "OK" if trace["selected_chunks"] else "EMPTY")

    def test_no_queries_no_matches_or_tiny_budget_returns_explicit_empty_evidence(self):
        index = RetrievalIndex(manuscript(), chunk_characters=180)
        cases = [([], {}), ([{"source": "test", "text": "unmatchablexylophone"}], {}),
                 ([{"source": "test", "text": "loan"}], {"budget_characters": 2})]
        for queries, kwargs in cases:
            with self.subTest(queries=queries, kwargs=kwargs):
                trace = index.retrieve(queries, **kwargs)
                self.assertEqual(trace["status"], "EMPTY")
                self.assertEqual(trace["selected_chunks"], [])
                self.assertEqual(trace["evidence_characters"], 0)
                self.assertEqual(trace["context_characters"], 2)
        self.assertEqual(RetrievalIndex("").retrieve([{"source": "test", "text": "loan"}])["status"], "EMPTY")

    def test_all_methods_are_selectable_and_hybrid_reports_both_rankings(self):
        index = RetrievalIndex(manuscript(), chunk_characters=180, embedding_backend=SemanticFixture())
        queries = [{"source": "issue:I000001:v1", "text": "loan"}]
        for method in ("lexical", "embedding", "hybrid"):
            with self.subTest(method=method):
                trace = index.retrieve(queries, method=method)
                self.assertEqual(trace["status"], "OK")
                self.assertEqual(trace["method"], method)
                self.assertEqual(trace["metadata"]["embedding_backend"], SemanticFixture.fingerprint)
                names = {r["method"] for r in trace["query_rankings"]}
                self.assertEqual(names, {"lexical", "embedding"} if method == "hybrid" else {method})
                self.assertIn("Loan", trace["selected_chunks"][0]["text"])
                self.assertEqual(trace, index.retrieve(queries, method=method))
        semantic = index.retrieve(queries, method="embedding")
        self.assertTrue(any("Mortgages" in c["text"] for c in semantic["selected_chunks"]))

    def test_equal_scores_have_stable_chunk_id_order(self):
        index = RetrievalIndex(manuscript(), chunk_characters=180, embedding_backend=SemanticFixture())
        trace = index.retrieve([{"source": "test", "text": "loan"}], method="embedding")
        ranking = trace["query_rankings"][0]["ranking"]
        self.assertGreaterEqual(len(ranking), 2)
        self.assertEqual(ranking[0]["score"], ranking[1]["score"])
        self.assertEqual([c["chunk_id"] for c in ranking], sorted(c["chunk_id"] for c in ranking))

    def test_near_duplicate_chunks_do_not_fill_the_evidence_budget(self):
        text = ("Repeated loan credit household evidence appears identically in this paragraph.\n\n" * 20)
        index = RetrievalIndex(text, chunk_characters=82)
        trace = index.retrieve([{"source": "test", "text": "loan credit"}])
        self.assertEqual(trace["status"], "OK")
        self.assertEqual(len(trace["selected_chunks"]), 1)

    def test_semantic_backend_is_explicit_and_failures_never_fall_back(self):
        queries = [{"source": "test", "text": "loan"}]
        for method in ("embedding", "hybrid"):
            with self.subTest(method=method), self.assertRaisesRegex(RetrievalError, "backend"):
                RetrievalIndex(manuscript()).retrieve(queries, method=method)

        class BrokenBackend:
            fingerprint = "broken-fixture-v1"

            def embed(self, texts):
                raise RuntimeError("Offline fixture backend failed")

        with self.assertRaisesRegex(RetrievalError, "backend failed"):
            RetrievalIndex(manuscript(), embedding_backend=BrokenBackend()).retrieve(queries, method="hybrid")

    def test_bad_embedding_vectors_and_missing_identity_fail_explicitly(self):
        queries = [{"source": "test", "text": "loan"}]
        factories = [lambda texts: [], lambda texts: [[float("nan")]] * len(texts),
                     lambda texts: [[float("inf")]] * len(texts), lambda texts: [[0.0]] * len(texts),
                     lambda texts: [[True]] * len(texts), lambda texts: [["1"]] * len(texts),
                     lambda texts: [[1.0] if i % 2 else [1.0, 0.0] for i in range(len(texts))]]
        for number, factory in enumerate(factories):
            backend = type("InvalidFixture", (), {"fingerprint": "invalid-v1", "embed": staticmethod(factory)})()
            with self.subTest(case=number), self.assertRaises(RetrievalError):
                RetrievalIndex(manuscript(), embedding_backend=backend).retrieve(queries, method="embedding")
        backend = SemanticFixture()
        backend.fingerprint = ""
        with self.assertRaisesRegex(RetrievalError, "fingerprint"):
            RetrievalIndex(manuscript(), embedding_backend=backend).retrieve(queries, method="embedding")

    def test_query_and_manuscript_embeddings_must_share_dimensions(self):
        class IncompatibleBackend:
            fingerprint = "incompatible-v1"

            def embed(self, texts):
                return [[1.0] if len(texts) == 1 else [1.0, 0.0] for _ in texts]

        with self.assertRaisesRegex(RetrievalError, "different dimensions"):
            RetrievalIndex(manuscript(), embedding_backend=IncompatibleBackend()).retrieve(
                [{"source": "test", "text": "loan"}], method="embedding")

    def test_invalid_retrieval_configuration_is_not_silently_reinterpreted(self):
        index = RetrievalIndex(manuscript())
        queries = [{"source": "test", "text": "loan"}]
        for kwargs in ({"method": "unknown"}, {"budget_characters": 1}, {"budget_characters": True},
                       {"max_manuscript_fraction": 0}, {"max_manuscript_fraction": 1},
                       {"max_manuscript_fraction": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RetrievalError):
                index.retrieve(queries, **kwargs)


class QueryTests(unittest.TestCase):
    def test_queries_use_current_active_claims_and_own_pre_with_auditable_sources(self):
        def concern(text):
            return {"technical_statement": text, "plain_language_statement": text + " explained",
                    "evidence": [{"locator": "Table 4", "support": text + " evidence"}]}

        snapshot = {
            "issue_views": [
                {"issue_id": "I000001", "version": 2, "active": True, "current": concern("currentclaim")},
                {"issue_id": "I000002", "version": 1, "active": False, "current": concern("retractedclaim")},
            ],
            "issues": [concern("historicaloriginal")], "issue_revisions": [concern("historicalrevision")],
            "arguments": [
                {"argument_id": "A000001", "issue_id": "I000001", "issue_version": 1,
                 "action": "CHALLENGE", "content": "oldversionargument", "evidence": []},
                {"argument_id": "A000002", "issue_id": "I000001", "issue_version": 2,
                 "action": "DEFENSE", "content": "currentdefense", "evidence": [{"support": "defenseevidence"}]},
                {"argument_id": "A000003", "issue_id": "I000002", "issue_version": 1,
                 "action": "CHALLENGE", "content": "retractedargument", "evidence": []},
                {"argument_id": "A000004", "issue_id": "I000001", "issue_version": 2,
                 "action": "QUESTION", "content": "questionwithoutclaim", "evidence": []},
                {"argument_id": "A000005", "issue_id": "I000001", "issue_version": 2,
                 "action": "CHALLENGE", "content": "currentchallenge", "evidence": []},
            ],
            "pre_assessments": [{"persona": "Other", "outcome": {"payload": {"revision_path": "peerprivatepre"}}}],
            "post_assessments": [],
        }
        own_pre = {"issues": [concern("ownpreconcern")],
                   "novelty_by_domain": {"empirical": {"rationale": "ownnovelty", "evidence": []}},
                   "insight": {"rationale": "owninsight"}, "repair_scope": {"rationale": "ownrepair"},
                   "revision_path": "ownrevision", "assessment_rationale": "ownassessment"}
        before = deepcopy((snapshot, own_pre))
        queries = build_queries(snapshot, own_pre)
        text = " ".join(q["text"] for q in queries)
        for included in ("currentclaim", "currentdefense", "defenseevidence", "currentchallenge",
                         "ownpreconcern", "ownnovelty", "owninsight", "ownrepair", "ownrevision", "ownassessment"):
            self.assertIn(included, text)
        for excluded in ("historicaloriginal", "historicalrevision", "retractedclaim", "oldversionargument",
                         "retractedargument", "questionwithoutclaim", "peerprivatepre"):
            self.assertNotIn(excluded, text)
        self.assertTrue(all(q["source"] and q["text"] for q in queries))
        self.assertEqual(len({q["text"] for q in queries}), len(queries))
        self.assertEqual((snapshot, own_pre), before)
        reordered = deepcopy(snapshot)
        reordered["issue_views"].reverse()
        reordered["arguments"].reverse()
        self.assertEqual(build_queries(reordered, own_pre), queries)

    def test_peer_post_is_rejected_before_any_query_is_built(self):
        for posts in ([{"persona": "Other", "outcome": {"status": "OK", "payload": {"secret": "peerpost"}}}],
                      [{"persona": "Other", "outcome": {"status": "FAILED", "payload": None}}]):
            with self.subTest(posts=posts), self.assertRaisesRegex(RetrievalError, "POST"):
                build_queries({"issue_views": [], "post_assessments": posts}, {})


if __name__ == "__main__":
    unittest.main()
