"""Attempt-level usage logging with explicit, user-supplied cost rates.

No hardcoded provider prices. Cache tokens are subsets of input and reasoning
tokens are subsets of output; neither is added to totals twice.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Union

from schemas import CallAttempt, NonnegativeNumber, Schema, Text, TokenUsage


class Pricing(Schema):
    """USD per million tokens under one explicitly documented billing regime."""
    input: NonnegativeNumber
    output: NonnegativeNumber
    cache_read: Optional[NonnegativeNumber] = None
    cache_write: Optional[NonnegativeNumber] = None
    basis: Text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TokenTracker:
    def __init__(self, pricing: Optional[Mapping[str, Union[Pricing, dict]]] = None):
        self.calls: list[CallAttempt] = []
        self.pricing = {
            model: rate if isinstance(rate, Pricing) else Pricing.model_validate(rate)
            for model, rate in (pricing or {}).items()
        }
        self.started_at = utc_now()

    def record_attempt(self, attempt: CallAttempt) -> None:
        """Preserve successful and failed attempts, including missing usage."""
        record = CallAttempt.model_validate(attempt.model_dump())
        if any(c.call_id == record.call_id and c.attempt == record.attempt for c in self.calls):
            raise ValueError("This call_id/attempt pair has already been recorded")
        self.calls.append(record)

    @staticmethod
    def _totals(calls: list[CallAttempt]) -> dict:
        def total(field):
            values = [getattr(c.usage, field) if c.usage else None for c in calls]
            return None if any(v is None for v in values) else sum(values)

        result = {
            "input": total("input_tokens"), "output": total("output_tokens"),
            "thinking": total("reasoning_tokens"), "cache_read": total("cache_read_tokens"),
            "cache_creation": total("cache_write_tokens"),
        }
        result["total"] = (
            None if result["input"] is None or result["output"] is None
            else result["input"] + result["output"]
        )
        return result

    def get_total_tokens(self) -> dict:
        return self._totals(self.calls)

    def _grouped(self, key) -> dict:
        groups = {}
        for call in self.calls:
            groups.setdefault(key(call), []).append(call)
        return {name: {**self._totals(calls), "calls": len(calls)} for name, calls in groups.items()}

    def get_tokens_by_role(self) -> dict:
        return self._grouped(lambda c: f"{c.stage}:{c.persona}" if c.persona else c.stage)

    def get_tokens_by_model(self) -> dict:
        return self._grouped(lambda c: c.reported_model or c.requested_model)

    def _call_cost(self, call: CallAttempt) -> Optional[float]:
        usage = call.usage
        rates = self.pricing.get(call.reported_model or call.requested_model)
        if usage is None or rates is None:
            return None
        quantities = [usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, usage.cache_write_tokens]
        if any(q is None for q in quantities):
            return None
        if usage.cache_read_tokens and rates.cache_read is None:
            return None
        if usage.cache_write_tokens and rates.cache_write is None:
            return None
        fresh = usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens
        return (
            fresh * rates.input + usage.output_tokens * rates.output
            + usage.cache_read_tokens * (rates.cache_read or 0)
            + usage.cache_write_tokens * (rates.cache_write or 0)
        ) / 1_000_000

    def estimate_cost(self, model: Optional[str] = None) -> Optional[float]:
        costs = [self._call_cost(c) for c in self.calls
                 if model is None or (c.reported_model or c.requested_model) == model]
        return None if any(c is None for c in costs) else sum(costs)

    def get_cost_by_model(self) -> dict:
        return {model: self.estimate_cost(model) for model in self.get_tokens_by_model()}

    def get_summary(self) -> dict:
        costs = [self._call_cost(c) for c in self.calls]
        return {
            "started_at": self.started_at,
            "total_calls": len({c.call_id for c in self.calls}),
            "total_attempts": len(self.calls),
            "retry_count": sum(c.retry_kind != "INITIAL" for c in self.calls),
            "total_tokens": self.get_total_tokens(),
            "by_role": self.get_tokens_by_role(), "by_model": self.get_tokens_by_model(),
            "cost_by_model": self.get_cost_by_model(), "total_cost": self.estimate_cost(),
            "known_cost_subtotal": sum(c for c in costs if c is not None),
            "unpriced_attempts": sum(c is None for c in costs),
        }

    def save_to_file(self, filename: Union[str, Path]) -> None:
        payload = {
            "summary": self.get_summary(),
            "pricing": {k: v.model_dump(mode="json") for k, v in self.pricing.items()},
            "attempts": [c.model_dump(mode="json") for c in self.calls],
        }
        Path(filename).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    def get_markdown_summary(self) -> str:
        summary = self.get_summary()
        cost = summary["total_cost"]
        label = "unavailable" if cost is None else f"${cost:.6f}"
        return (
            f"Calls: {summary['total_calls']}; attempts: {summary['total_attempts']}\n\n"
            f"Total tokens: {summary['total_tokens']['total']}\n\nEstimated cost: {label}"
        )

    def print_summary(self) -> None:
        print(self.get_markdown_summary())
