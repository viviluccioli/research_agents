"""Deterministic manuscript retrieval with exact, auditable source slices.

No provider is loaded here. Optional semantic backends supply a stable ``fingerprint``
and ``embed(texts)`` method; selecting semantic retrieval never downloads a model or
silently falls back to another method.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from numbers import Real
import re
from typing import Protocol


RETRIEVAL_VERSION = "exp12-retrieval-v1"
_STOP_WORDS = frozenset("a an and are as at be been by for from has have in is it its of on or that the their this to was were will with".split())
_RRF_CONSTANT = 60
_REDUNDANCY_LIMIT = 0.85


class RetrievalError(RuntimeError):
    """Evidence retrieval failed; callers must preserve the failure explicitly."""


class EmbeddingBackend(Protocol):
    fingerprint: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


def _compact(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"\w+", text.casefold()) if word not in _STOP_WORDS]


def build_queries(final_snapshot: dict, own_pre: dict) -> list[dict]:
    """Use existing claims/evidence only; no peer POST output or generated queries."""
    if final_snapshot.get("post_assessments"):
        raise RetrievalError("Retrieval requires a snapshot without POST assessments")
    queries, seen = [], set()

    def add(source, text):
        if isinstance(text, str):
            text = " ".join(text.split())
            if text and text not in seen:
                seen.add(text)
                queries.append({"source": source, "text": text})

    def evidence(source, records):
        for index, item in enumerate(records or []):
            text = " ".join(value for value in (item.get("locator"), item.get("support")) if value)
            add(f"{source}:evidence:{index}", text)

    def concern(source, item):
        for field in ("technical_statement", "plain_language_statement"):
            add(f"{source}:{field}", item.get(field))
        evidence(source, item.get("evidence"))

    active = {view["issue_id"]: view for view in final_snapshot.get("issue_views", []) if view["active"]}
    for issue_id, view in sorted(active.items()):
        concern(f"issue:{issue_id}:v{view['version']}", view["current"])
    for argument in sorted(final_snapshot.get("arguments", []), key=lambda item: item["argument_id"]):
        view = active.get(argument["issue_id"])
        if (view and argument["issue_version"] == view["version"]
                and argument["action"] in ("CHALLENGE", "DEFENSE")):
            source = f"argument:{argument['argument_id']}"
            add(source, argument["content"])
            evidence(source, argument.get("evidence"))
    for index, item in enumerate(own_pre.get("issues", [])):
        concern(f"own_pre:issue:{index}", item)
    for domain, judgment in sorted(own_pre.get("novelty_by_domain", {}).items()):
        source = f"own_pre:novelty:{domain}"
        add(source, judgment.get("rationale"))
        evidence(source, judgment.get("evidence"))
    for field in ("insight", "repair_scope"):
        judgment = own_pre.get(field, {})
        source = f"own_pre:{field}"
        add(source, judgment.get("rationale"))
        evidence(source, judgment.get("evidence"))
    for field in ("revision_path", "assessment_rationale"):
        add(f"own_pre:{field}", own_pre.get(field))
    return queries


def chunk_manuscript(manuscript: str, chunk_characters: int = 1600) -> list[dict]:
    """Cover the source exactly, preferring paragraph, sentence, then word breaks.

    A smaller target on short documents permits selecting whole chunks under the
    default quarter-manuscript evidence cap. Boundaries never depend on section names.
    """
    if not isinstance(manuscript, str):
        raise TypeError("Manuscript must be text")
    if type(chunk_characters) is not int or chunk_characters < 1:
        raise ValueError("chunk_characters must be a positive integer")
    target = min(chunk_characters, max(1, len(manuscript) // 8))
    chunks, start = [], 0
    while start < len(manuscript):
        end = min(len(manuscript), start + target)
        if end < len(manuscript):
            window = manuscript[start:end]
            lower = max(1, len(window) // 2)
            # Regexes describe mechanical whitespace/punctuation, never headings.
            for pattern in (r"\n\s*\n", r"[.!?](?=\s)", r"\s+"):
                boundaries = [match.end() for match in re.finditer(pattern, window) if match.end() >= lower]
                if boundaries:
                    end = start + boundaries[-1]
                    break
        chunks.append({"chunk_id": f"C{len(chunks) + 1:06d}", "start": start, "end": end,
                       "characters": end - start, "text": manuscript[start:end]})
        start = end
    for index, chunk in enumerate(chunks):
        chunk["previous_chunk_id"] = chunks[index - 1]["chunk_id"] if index else None
        chunk["next_chunk_id"] = chunks[index + 1]["chunk_id"] if index + 1 < len(chunks) else None
    return chunks


class RetrievalIndex:
    def __init__(self, manuscript: str, *, chunk_characters: int = 1600, embedding_backend=None):
        self.manuscript = manuscript
        self.chunk_characters = chunk_characters
        self.chunks = chunk_manuscript(manuscript, chunk_characters)
        self.embedding_backend = embedding_backend
        self._chunk_vectors = None
        self._counts = [Counter(_tokens(chunk["text"])) for chunk in self.chunks]
        self._lengths = [sum(counts.values()) for counts in self._counts]
        self._average_length = sum(self._lengths) / max(1, len(self.chunks)) or 1.0
        self._document_frequency = Counter(word for counts in self._counts for word in counts)

    def metadata(self) -> dict:
        fingerprint = None
        if self.embedding_backend is not None:
            fingerprint = getattr(self.embedding_backend, "fingerprint", None)
            if not isinstance(fingerprint, str) or not fingerprint.strip():
                raise RetrievalError("Embedding backend requires a stable nonempty fingerprint")
        return {"retrieval_version": RETRIEVAL_VERSION, "chunk_characters": self.chunk_characters,
                "effective_chunk_characters": min(self.chunk_characters, max(1, len(self.manuscript) // 8)),
                "embedding_backend": fingerprint, "bm25_k1": 1.2, "bm25_b": 0.75,
                "rank_fusion_constant": _RRF_CONSTANT, "redundancy_jaccard_limit": _REDUNDANCY_LIMIT,
                "chunk_overlap": 0,
                "tokenization": "unicode_word_casefold_fixed_stop_words_v1"}

    def _lexical_scores(self, query: str) -> list[float]:
        terms = set(_tokens(query))
        scores = []
        for counts, length in zip(self._counts, self._lengths):
            score = 0.0
            for word in sorted(terms):
                count = counts[word]
                if count:
                    frequency = self._document_frequency[word]
                    idf = math.log(1 + (len(self.chunks) - frequency + 0.5) / (frequency + 0.5))
                    score += idf * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * length / self._average_length))
            scores.append(score)
        return scores

    def _embed(self, texts: list[str]) -> list[tuple]:
        if self.embedding_backend is None:
            raise RetrievalError("Embedding/hybrid retrieval requires an explicitly supplied embedding backend")
        self.metadata()  # Validate the identity that will accompany semantic results.
        try:
            raw = list(self.embedding_backend.embed(texts))
        except Exception as exc:
            raise RetrievalError(f"Embedding backend failed ({type(exc).__name__})") from exc
        if len(raw) != len(texts):
            raise RetrievalError("Embedding backend returned the wrong number of vectors")
        vectors, dimension = [], None
        for row in raw:
            try:
                values = list(row)
            except TypeError as exc:
                raise RetrievalError("Embedding vector must be a sequence") from exc
            if not values or any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
                raise RetrievalError("Embedding vectors require nonempty numeric dimensions")
            vector = tuple(float(value) for value in values)
            if not all(math.isfinite(value) for value in vector):
                raise RetrievalError("Embedding vectors must contain finite values")
            if dimension is not None and len(vector) != dimension:
                raise RetrievalError("Embedding vectors have inconsistent dimensions")
            dimension = len(vector)
            norm = math.sqrt(sum(value * value for value in vector))
            if not math.isfinite(norm) or norm == 0:
                raise RetrievalError("Embedding vectors require a finite nonzero norm")
            vectors.append(tuple(value / norm for value in vector))
        return vectors

    def retrieve(self, queries: list[dict], *, method: str = "lexical", budget_characters: int = 12000,
                 max_manuscript_fraction: float = 0.25) -> dict:
        """Budget the serialized evidence array, including every sent provenance field.

        Rank fusion treats each query/method as a separate ranked list. Greedy
        selection skips near-duplicate word sets and keeps whole source chunks.
        A separate raw-text cap prevents reconstructing a short manuscript.
        """
        if method not in ("lexical", "embedding", "hybrid"):
            raise RetrievalError(f"Unknown retrieval method: {method}")
        if type(budget_characters) is not int or budget_characters < 2:
            raise RetrievalError("Evidence budget must accommodate at least the empty JSON array (2 characters)")
        if (isinstance(max_manuscript_fraction, bool) or not isinstance(max_manuscript_fraction, Real)
                or not math.isfinite(max_manuscript_fraction) or not 0 < max_manuscript_fraction < 1):
            raise RetrievalError("Manuscript evidence fraction must be finite and strictly between zero and one")
        if not isinstance(queries, list) or any(not isinstance(query, dict)
                or not isinstance(query.get("source"), str) or not isinstance(query.get("text"), str)
                or not query["text"].strip() for query in queries):
            raise RetrievalError("Queries must contain source and nonempty text strings")
        metadata = self.metadata()
        if method in ("embedding", "hybrid") and self.embedding_backend is None:
            raise RetrievalError("Embedding/hybrid retrieval requires an explicitly supplied embedding backend")
        trace = {"status": "EMPTY", "queries": deepcopy(queries), "selected_chunks": [],
                 "context_characters": 2, "evidence_characters": 0, "method": method,
                 "budget_characters": budget_characters, "metadata": metadata,
                 "max_manuscript_fraction": max_manuscript_fraction,
                 "evidence_limit_characters": min(budget_characters, math.floor(len(self.manuscript) * max_manuscript_fraction))}
        if not queries or not self.chunks:
            return trace
        semantic_scores = None
        if method in ("embedding", "hybrid"):
            if self._chunk_vectors is None:
                self._chunk_vectors = self._embed([chunk["text"] for chunk in self.chunks])
            query_vectors = self._embed([query["text"] for query in queries])
            if len(query_vectors[0]) != len(self._chunk_vectors[0]):
                raise RetrievalError("Query and manuscript embeddings have different dimensions")
            semantic_scores = [[sum(a * b for a, b in zip(query_vector, chunk_vector))
                                for chunk_vector in self._chunk_vectors] for query_vector in query_vectors]
        method_scores = {name: [0.0] * len(self.chunks) for name in
                         (("lexical", "embedding") if method == "hybrid" else (method,))}
        # Preserve detailed query rankings outside the budgeted/sent evidence array.
        trace["query_rankings"] = []
        for query_index, query in enumerate(queries):
            rankings = {}
            if method in ("lexical", "hybrid"):
                rankings["lexical"] = self._lexical_scores(query["text"])
            if semantic_scores is not None:
                rankings["embedding"] = semantic_scores[query_index]
            for retrieval_method, scores in sorted(rankings.items()):
                ranked = sorted((index for index, score in enumerate(scores) if score > 0),
                                key=lambda index: (-scores[index], self.chunks[index]["chunk_id"]))
                trace["query_rankings"].append({"query_index": query_index, "method": retrieval_method,
                    "ranking": [{"chunk_id": self.chunks[index]["chunk_id"], "score": scores[index], "rank": rank}
                                for rank, index in enumerate(ranked, 1)]})
                for rank, index in enumerate(ranked, 1):
                    method_scores[retrieval_method][index] += 1 / (_RRF_CONSTANT + rank)
        scores_by_chunk = [{} for _ in self.chunks]
        fused_scores = [0.0] * len(self.chunks)
        for retrieval_method, scores in sorted(method_scores.items()):
            ranked = sorted((index for index, score in enumerate(scores) if score > 0),
                            key=lambda index: (-scores[index], self.chunks[index]["chunk_id"]))
            for rank, index in enumerate(ranked, 1):
                scores_by_chunk[index][retrieval_method] = {"score": scores[index], "rank": rank}
                fused_scores[index] += 1 / (_RRF_CONSTANT + rank)
        ranked_chunks = sorted((index for index, score in enumerate(fused_scores) if score > 0),
                               key=lambda index: (-fused_scores[index], self.chunks[index]["chunk_id"]))
        selected_sets = []
        for retrieval_rank, index in enumerate(ranked_chunks, 1):
            chunk = self.chunks[index]
            words = set(self._counts[index])
            if any(len(words & previous) / max(1, len(words | previous)) >= _REDUNDANCY_LIMIT
                   for previous in selected_sets):
                continue
            if trace["evidence_characters"] + chunk["characters"] > trace["evidence_limit_characters"]:
                continue
            selected = {**deepcopy(chunk), "method_scores": scores_by_chunk[index],
                        "fused_score": fused_scores[index], "retrieval_rank": retrieval_rank,
                        "selection_rank": len(trace["selected_chunks"]) + 1}
            proposed = trace["selected_chunks"] + [selected]
            size = len(_compact(proposed))
            if size > budget_characters:
                continue
            trace["selected_chunks"] = proposed
            trace["context_characters"] = size
            trace["evidence_characters"] += chunk["characters"]
            selected_sets.append(words)
        trace["status"] = "OK" if trace["selected_chunks"] else "EMPTY"
        return trace
