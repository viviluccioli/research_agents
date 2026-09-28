"""Discussion scaffold for comparing configurations of the frozen MAD architecture.

The eventual goal is external validity and review quality, not an internal
paper-quality score. Candidate architecture parameters include debate rounds,
POST manuscript-context strategy, retrieval method and evidence budget, possibly
panel size, and other explicitly defined architecture information-flow choices.
Listing a parameter here does not implement or authorize a change to exp-12.

Expected inputs: raw run records and configuration/provenance, paper identifiers,
external outcomes, and an explicitly agreed evaluation/splitting specification.
Expected outputs: paper-level split assignments, outcome-specific architecture
comparisons, uncertainty summaries, and separate efficiency/cost diagnostics.
TODO: agree input locations, joins, representations and output formats. Function
arguments below describe responsibilities, not finalized interfaces.

This module only documents future work. It loads no data and performs no analysis.
"""

# Shared design principles
# - The 50 papers are the independent units. Keep all runs, versions and reviews
#   for a paper together in cross-validation, bootstrapping and permutation.
# - The 1,225 unordered pairs derived from 50 papers are not 1,225 independent
#   observations. Construct comparisons within the agreed paper-level design.
# - Keep metrics and external outcomes separate rather than collapsing them early.
# - Journal-ranking systems are correlated robustness measures of a similar
#   signal, not independent replications; do not combine them by default.
# - Working-paper and published-version citations remain separate outcomes.
# - Keep PRE, POST and debate-mechanism quantities distinguishable.
# - Feature importance denotes predictive association unless a separate causal
#   design justifies stronger language.
# - Track failures, retries, truncation and missing calls separately from
#   substantive paper-quality variables.

# 1. Inputs and provenance
# TODO: agree run inclusion, paper linkage and external-outcome definitions,
# including citation measurement dates and treatment of unavailable outcomes.
def load_architecture_runs(source):
    """Eventually return paper-linked raw runs and architecture provenance."""
    raise NotImplementedError("TODO: agree run-loading and linkage contracts.")


def load_external_outcomes(source):
    """Eventually return distinct paper-linked outcomes with source metadata."""
    raise NotImplementedError("TODO: agree external-outcome loading contracts.")


# 2. Paper-level design and uncertainty
# TODO: agree split/resampling schemes, tuning/evaluation separation, treatment
# of repeated runs, and uncertainty reporting without leakage between papers.
def create_paper_level_splits(paper_ids, split_spec):
    """Eventually return paper-level assignments under an agreed split design."""
    raise NotImplementedError("TODO: agree the paper-level split specification.")


def bootstrap_by_paper(records, bootstrap_spec):
    """Eventually return paper-resampled summaries under an agreed specification."""
    raise NotImplementedError("TODO: agree bootstrap units, estimands and intervals.")


# 3. Candidate evaluation targets (no primary metric selected)
# - Pairwise ordering success; Kendall and Spearman rank correlation.
# - Performance against each individual journal-ranking system.
# - Citation outcomes, with working-paper and published versions separate.
# - Human economist evaluations of review quality.
# - Efficiency/cost where relevant, reported separately from scholarly quality.
# TODO: agree metrics, primary target, tie handling, missing-data treatment,
# aggregation and any weighting. Deriving an ordering from raw review outputs
# also requires an explicit downstream specification; none is supplied here.
def evaluate_architecture(runs, outcomes, paper_splits, evaluation_spec):
    """Eventually return separate metric/outcome results for one configuration."""
    raise NotImplementedError("TODO: agree architecture evaluation rules.")


def evaluate_pairwise_ordering(paper_outputs, outcomes, paper_splits, ordering_spec):
    """Eventually return ordering results that retain paper-pair provenance."""
    raise NotImplementedError("TODO: agree ordering, ties and paper-level inference.")


# 4. Architectural comparisons
# - PRE versus POST; debate versus no debate; one versus multiple rounds.
# - Abstract-only versus retrieved POST context.
# - MAD versus relevant baseline architectures.
# TODO: define matched comparisons, weighting/aggregation, architecture-selection
# rules and safeguards separating selection from held-out evaluation. No search
# grid, optimizer or preferred configuration is specified here.
def compare_architectures(evaluation_results, comparison_spec):
    """Eventually return configuration comparisons under an agreed analysis plan."""
    raise NotImplementedError("TODO: agree architecture comparison and selection rules.")
