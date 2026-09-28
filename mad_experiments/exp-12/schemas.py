"""exp-12 contracts. No model calls, graph mutation, or paper-level scoring.

JSON Schema is generated from these models. Structural validation lives here;
snapshot visibility, ownership and reference checks live in exp-12.py.
"""
from __future__ import annotations

from typing import Annotated, Generic, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SCHEMA_VERSION = "exp12-v2.1"
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
IssueId = Annotated[str, StringConstraints(pattern=r"^I[0-9]{6,}$")]
ArgumentId = Annotated[str, StringConstraints(pattern=r"^A[0-9]{6,}$")]
NonnegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
Severity = Annotated[float, Field(ge=1, le=10)]
Confidence = Annotated[float, Field(ge=0, le=10)]
NonnegativeNumber = Annotated[float, Field(ge=0)]
Persona = Literal[
    "Theorist", "Econometrician", "AI_Expert", "Data_Scientist", "CS_Expert",
    "Visionary", "Policymaker", "Ethicist", "Perspective", "Historian",
]
Verdict = Literal["ACCEPT", "RESUBMIT", "REJECT"]
NoveltyDomain = Literal["methodological", "empirical", "theoretical", "policy"]
IssueType = Literal[
    "substantive_flaw", "substantial_rephrasal", "rephrasal", "extension",
    "literature_dispute", "insight_weakness", "future_work",
]
Stance = Literal["MAINTAIN", "NARROW", "RETRACT", "UNCERTAIN"]
Stage = Literal["SELECTION", "PRE", "DEBATE", "POST", "EDITOR"]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def distinct(values: list, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be distinct")


class Evidence(Schema):
    kind: Literal["MANUSCRIPT", "OMISSION"]
    locator: Optional[Text]
    support: Text


class AssessedJudgment(Schema):
    level: Literal["HIGH", "MEDIUM", "LOW", "NONE"]
    rationale: Text
    evidence: Annotated[list[Evidence], Field(min_length=1)]


class AbstainedJudgment(Schema):
    level: Literal["ABSTAIN"]
    reason: Text


DomainJudgment = Annotated[
    Union[AssessedJudgment, AbstainedJudgment], Field(discriminator="level")
]
# Shared representation, different construct definitions in the review protocol.
InsightAssessment = DomainJudgment


class NoveltyByDomain(Schema):
    methodological: DomainJudgment
    empirical: DomainJudgment
    theoretical: DomainJudgment
    policy: DomainJudgment


class AssessedRepair(Schema):
    status: Literal["ASSESSED"]
    level: Annotated[int, Field(ge=0, le=4)]
    rationale: Text


class AbstainedRepair(Schema):
    status: Literal["ABSTAIN"]
    reason: Text


RepairScope = Annotated[Union[AssessedRepair, AbstainedRepair], Field(discriminator="status")]


class SelectionOutput(Schema):
    selected_personas: Annotated[list[Persona], Field(min_length=3, max_length=3)]
    selection_rationale: Text

    @model_validator(mode="after")
    def unique_panel(self):
        distinct(self.selected_personas, "selected_personas")
        return self


class Assessment(Schema):
    """Identical construct fields for PRE and POST; no aggregation."""
    novelty_by_domain: NoveltyByDomain
    insight: InsightAssessment
    repair_scope: RepairScope
    revision_path: Text
    verdict: Verdict
    assessment_rationale: Text


class IssueCandidate(Schema):
    issue_type: IssueType
    # Empty when this is not a novelty concern; multiple domains are possible.
    novelty_domains: list[NoveltyDomain]
    technical_statement: Text
    plain_language_statement: Text
    severity: Severity
    severity_rationale: Text
    initial_confidence: Confidence
    evidence: Annotated[list[Evidence], Field(min_length=1)]
    related_issue_ids: list[IssueId]

    @model_validator(mode="after")
    def unique_links(self):
        distinct(self.novelty_domains, "novelty_domains")
        distinct(self.related_issue_ids, "related_issue_ids")
        return self


class PreAssessment(Assessment):
    structural_strength: Text
    best_case_contribution: Text
    issues: list[IssueCandidate]

    @model_validator(mode="after")
    def independent_issues(self):
        if any(issue.related_issue_ids for issue in self.issues):
            raise ValueError("PRE cannot reference unseen peer issue IDs")
        return self


class ArgumentCandidate(Schema):
    issue_id: IssueId
    responds_to_argument_id: Optional[ArgumentId]
    action: Literal["CHALLENGE", "DEFENSE", "CONCESSION", "QUESTION"]
    content: Text
    evidence: list[Evidence]
    confidence: Optional[Confidence]

    @model_validator(mode="after")
    def claim_support(self):
        if self.action in ("CHALLENGE", "DEFENSE"):
            if self.confidence is None or not self.evidence:
                raise ValueError("CHALLENGE/DEFENSE require confidence and evidence")
        elif self.confidence is not None:
            raise ValueError("QUESTION/CONCESSION must have null confidence")
        return self


class OriginatorUpdate(Schema):
    issue_id: IssueId
    stance: Stance
    rationale: Text
    confidence: Optional[Confidence]
    revised_issue: Optional[IssueCandidate]

    @model_validator(mode="after")
    def stance_fields(self):
        if self.stance == "RETRACT":
            if self.confidence is not None or self.revised_issue is not None:
                raise ValueError("RETRACT must have null confidence and no revision")
        elif self.confidence is None:
            raise ValueError("A claim-bearing stance requires confidence")
        if self.stance == "NARROW":
            if self.revised_issue is None:
                raise ValueError("NARROW requires a revised issue")
        if self.revised_issue is not None:
            if self.confidence != self.revised_issue.initial_confidence:
                raise ValueError("Update confidence must match the revised proposition")
            if self.issue_id in self.revised_issue.related_issue_ids:
                raise ValueError("A revision cannot link to itself")
        return self


class LinkProposal(Schema):
    issue_ids: Annotated[list[IssueId], Field(min_length=2, max_length=2)]
    relationship: Literal["RELATED", "POSSIBLE_DUPLICATE"]
    rationale: Text

    @model_validator(mode="after")
    def distinct_endpoints(self):
        distinct(self.issue_ids, "issue_ids")
        return self


class DebateOutput(Schema):
    arguments: list[ArgumentCandidate]
    new_issues: list[IssueCandidate]
    originator_updates: list[OriginatorUpdate]
    issue_links: list[LinkProposal]

    @model_validator(mode="after")
    def unique_updates(self):
        distinct([u.issue_id for u in self.originator_updates], "updated issue IDs")
        return self


class PostAssessment(Assessment):
    originator_updates: list[OriginatorUpdate]
    late_observations: list[IssueCandidate]

    @model_validator(mode="after")
    def unique_updates(self):
        distinct([u.issue_id for u in self.originator_updates], "updated issue IDs")
        return self


class EditorOutput(Schema):
    integrated_rationale: Text
    verdict: Verdict
    author_letter: Text
    revision_path: Text
    constructive_summary: Text
    decisive_issue_ids: list[IssueId]

    @model_validator(mode="after")
    def unique_issues(self):
        distinct(self.decisive_issue_ids, "decisive_issue_ids")
        return self


# Engine-authored records. Timing + stance events replace procedural_state,
# late flags, mutable current confidence and duplicate final stance fields.
class EventSource(Schema):
    source_call_id: Text
    event_sequence: PositiveInt


class Issue(IssueCandidate, EventSource):
    issue_id: IssueId
    originator: Persona
    phase_created: Literal["PRE", "DEBATE", "POST"]
    round_created: NonnegativeInt

    @model_validator(mode="after")
    def creation_fields(self):
        if (self.phase_created == "PRE") != (self.round_created == 0):
            raise ValueError("Only PRE issues use round zero")
        if self.issue_id in self.related_issue_ids:
            raise ValueError("An issue cannot link to itself")
        if self.phase_created == "PRE" and self.related_issue_ids:
            raise ValueError("Independent PRE issues cannot reference other issues")
        return self


class IssueRevision(IssueCandidate, EventSource):
    issue_id: IssueId
    version: Annotated[int, Field(ge=2)]
    persona: Persona
    phase: Literal["DEBATE", "POST"]
    round_number: PositiveInt


class IssueUpdate(EventSource):
    """Append-only judgment on an issue; original claim text is never overwritten."""
    issue_id: IssueId
    persona: Persona
    phase: Literal["DEBATE", "POST"]
    round_number: PositiveInt
    stance: Stance
    rationale: Text
    confidence: Optional[Confidence]
    issue_version: PositiveInt

    @model_validator(mode="after")
    def stance_fields(self):
        if (self.stance == "RETRACT") != (self.confidence is None):
            raise ValueError("Only RETRACT has null confidence")
        return self


class Argument(ArgumentCandidate, EventSource):
    issue_version: PositiveInt
    argument_id: ArgumentId
    persona: Persona
    round_created: PositiveInt


class IssueLink(LinkProposal, EventSource):
    proposer: Persona
    round_created: PositiveInt


T = TypeVar("T")


class Outcome(Schema, Generic[T]):
    status: Literal["OK", "FAILED", "SKIPPED", "FALLBACK"]
    payload: Optional[T]
    call_ids: list[Text]
    reason: Optional[Text]

    @model_validator(mode="after")
    def availability(self):
        distinct(self.call_ids, "call_ids")
        if self.status in ("OK", "FALLBACK"):
            if self.payload is None:
                raise ValueError("Successful outcomes require a payload")
        elif self.payload is not None or self.reason is None:
            raise ValueError("Failed/skipped outcomes require null payload and a reason")
        if self.status == "FALLBACK":
            if not isinstance(self.payload, SelectionOutput) or self.reason is None:
                raise ValueError("Only selection may use a fallback; reason is required")
        return self


class AssessmentRecord(Schema, Generic[T]):
    persona: Persona
    outcome: Outcome[T]


class ContextArtifact(Schema):
    path: Text
    sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    characters: NonnegativeInt
    manuscript_characters: NonnegativeInt
    truncated: bool
    extraction_method: Optional[Text]


class TokenUsage(Schema):
    """Normalized totals: cache is within input; reasoning is within output.

    Missing usage is null, not zero. Provider-specific normalization belongs
    in the future call adapter and must occur before this record is created.
    """
    input_tokens: Optional[NonnegativeInt]
    output_tokens: Optional[NonnegativeInt]
    reasoning_tokens: Optional[NonnegativeInt]
    cache_read_tokens: Optional[NonnegativeInt]
    cache_write_tokens: Optional[NonnegativeInt]

    @model_validator(mode="after")
    def subsets(self):
        if self.input_tokens is not None:
            cache = (self.cache_read_tokens or 0) + (self.cache_write_tokens or 0)
            if cache > self.input_tokens:
                raise ValueError("Cache subsets cannot exceed normalized input tokens")
        if self.output_tokens is not None and self.reasoning_tokens is not None:
            if self.reasoning_tokens > self.output_tokens:
                raise ValueError("Reasoning tokens cannot exceed normalized output tokens")
        return self


class CallAttempt(Schema):
    call_id: Text
    stage: Stage
    persona: Optional[Persona]
    round_number: NonnegativeInt
    attempt: PositiveInt
    retry_kind: Literal["INITIAL", "TRANSPORT", "VALIDATION"]
    requested_model: Text
    reported_model: Optional[Text]
    requested_temperature: Optional[NonnegativeNumber]
    sent_temperature: Optional[NonnegativeNumber]
    reported_temperature: Optional[NonnegativeNumber]
    reasoning_requested: Optional[Text]
    reasoning_sent: Optional[Text]
    request_context: Optional[ContextArtifact]
    raw_response_path: Optional[Text]
    status: Literal["OK", "TRANSPORT_ERROR", "PARSE_ERROR", "SCHEMA_ERROR", "GRAPH_ERROR"]
    errors: list[Text]
    started_at: Text
    elapsed_seconds: Optional[NonnegativeNumber]
    usage: Optional[TokenUsage]


class Opportunity(Schema):
    issue_id: IssueId
    issue_version: PositiveInt
    persona: Persona
    round_number: PositiveInt
    call_id: Text
    outcome: Literal["VALID_RESPONSE", "FAILED_CALL"]
    response_argument_ids: list[ArgumentId]


class RoundRecord(Schema):
    round_number: PositiveInt
    snapshot: ContextArtifact
    responses: list[AssessmentRecord[DebateOutput]]
    opportunities: list[Opportunity]


class RunResult(Schema):
    architecture_version: Text
    schema_version: Text
    prompt_version: Text
    run_id: Text
    status: Literal["COMPLETE", "PARTIAL", "FAILED"]
    code_commit: Optional[Text]
    code_sha256: Text
    prompt_sha256: Text
    settings_sha256: Text
    schema_sha256: Text
    manuscript_sha256: Text
    manuscript_characters: NonnegativeInt
    started_at: Text
    finished_at: Text
    rounds_requested: NonnegativeInt
    abstract: Optional[ContextArtifact]
    post_context_artifacts: list[ContextArtifact]
    selection: Outcome[SelectionOutput]
    pre_assessments: list[AssessmentRecord[PreAssessment]]
    issues: list[Issue]
    issue_revisions: list[IssueRevision]
    issue_updates: list[IssueUpdate]
    arguments: list[Argument]
    issue_links: list[IssueLink]
    rounds: list[RoundRecord]
    post_assessments: list[AssessmentRecord[PostAssessment]]
    editor: Outcome[EditorOutput]
    calls: list[CallAttempt]
    report_path: Optional[Text]
    diagnostic_messages: list[Text]

    @model_validator(mode="after")
    def assessment_roster(self):
        selected = self.selection.payload
        expected = set(selected.selected_personas) if selected else set()
        for records in (self.pre_assessments, self.post_assessments):
            distinct([r.persona for r in records], "assessment personas")
            if {r.persona for r in records} != expected:
                raise ValueError("Assessment records must cover exactly the selected panel")
        return self


MODEL_OUTPUT_SCHEMAS = {
    "SELECTION": SelectionOutput, "PRE": PreAssessment, "DEBATE": DebateOutput,
    "POST": PostAssessment, "EDITOR": EditorOutput,
}


def output_json_schema(stage: Stage) -> dict:
    """Local source of truth; no provider-specific schema compatibility assumed."""
    return MODEL_OUTPUT_SCHEMAS[stage].model_json_schema()
