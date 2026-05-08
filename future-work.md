# Future Work & Directions

## Overview

This document outlines the planned future directions for the Research Agents project, including benchmarking experiments, ongoing maintenance needs, and integration with the Board AI program infrastructure.

## 1. Benchmarking Experiment

### Current Status

**Ground Truth Dataset Assembled**: The `experiment/` directory contains papers organized into **4 tiers** based on real-world publication outcomes:

- **Tier 1**: Papers accepted to **top-tier journals** (AER, QJE, JPE, Econometrica, etc.)
- **Tier 2**: Papers accepted to **mid-tier journals** or specialized venues
- **Tier 3**: Papers published as **FEDS Notes or IFDP notes** (Federal Reserve internal publications)
- **Tier 4**: Papers **never published externally** (working papers, unpublished manuscripts)

**Status**: Papers collected, awaiting debate engine finalization before running full benchmark.

### Experiment Objectives

1. **Accuracy Validation**: Measure how well the MAD system's verdicts correlate with real-world editorial decisions
   - Does the system ACCEPT Tier 1 papers at higher rates than Tier 4?
   - What is the false positive rate (accepting Tier 4 papers)?
   - What is the false negative rate (rejecting Tier 1 papers)?

2. **Calibration Analysis**: Understand if verdict confidence aligns with ground truth quality
   - Are weighted consensus scores predictive of publication tier?
   - Do persona selection patterns differ across tiers?

3. **Qualitative Evaluation**: Review debate transcripts to understand reasoning quality
   - Are critiques substantive and accurate?
   - Do personas identify the same issues as human referees?
   - Are there systematic blind spots?

### Experiment Workflow

**Once debate engine finalized**:

1. **Batch Processing**: Run `experiment/batch_referee_reports.py` on all papers in ground truth dataset
   ```bash
   cd experiment
   python batch_referee_reports.py \
       --pdf-dir papers/ \
       --ground-truth tracking.csv \
       --output-dir results/benchmark_YYYYMMDD/
   ```

2. **Results Analysis**: Compute accuracy metrics
   - Overall accuracy by tier
   - Confusion matrix (predicted verdict × true tier)
   - Precision/recall for ACCEPT vs. REJECT
   - Cost analysis (tokens × papers)

3. **Qualitative Review**: Sample debate transcripts for manual evaluation
   - Select 10-20 papers across tiers
   - Compare MAD critiques to actual referee reports (if available)
   - Identify systematic strengths/weaknesses

4. **Iteration**: Based on findings, refine prompts/personas/debate structure and re-run

### Open Questions

- **Verdict mapping**: How do we map ACCEPT/REVISE/REJECT to 4-tier ground truth? (Binary: ACCEPT vs. not? Or ordinal scale?)
- **Baseline comparison**: Should we compare against simpler baselines (single-LLM evaluation, section evaluator only)?
- **Paper type distribution**: Do we have balanced representation across empirical/theoretical/policy papers in each tier?

## 2. Ongoing Maintenance & Development

### App Maintenance

**Core Responsibilities**:
- Monitor API usage and costs (track token consumption via `utils.py` logging)
- Update Claude model versions as Anthropic releases new models
- Respond to user bug reports and feature requests
- Maintain prompt versions (`prompts/*/config.yaml`)

**Regular Tasks**:
- **Weekly**: Review Streamlit app logs for errors
- **Monthly**: Check cache directory size (`.referee_cache/` can grow large)
- **Quarterly**: Audit prompt effectiveness, consider versioning updates

### Feature Development

**Near-term enhancements**:
1. **Persona selection consistency**: Continue monitoring Round 0 variability (see `persona_exp/`)
2. **Quote validation improvements**: Tune thresholds based on false positive/negative rates
3. **Export formats**: Add Word/Google Docs export (currently MD/CSV/PDF)
4. **User feedback loop**: Add UI for users to rate referee report quality

**Medium-term enhancements**:
1. **Custom persona creation**: Allow users to define ad-hoc personas for specialized domains
2. **Comparative evaluation**: Side-by-side comparison of multiple papers
3. **Longitudinal tracking**: Store evaluation history per paper (revision tracking)
4. **Fine-tuned personas**: Explore fine-tuning smaller models for specific personas

### Code Quality

**Testing**:
- Maintain test coverage in `app_system/tests/`
- Add regression tests for consistency improvements (Phase 1, Phase 2)
- Automated testing on commit (consider CI/CD integration)

**Documentation**:
- Keep `CLAUDE.md` updated as architecture evolves
- Document major changes in `commit_history/` (auto-generated via hooks)
- Update `handoff_context.md` as team members change

## 3. Board AI Integration

### Vision

The Research Agents system will be integrated into a **unified Board AI platform** alongside other AI tools, starting with the **MarginalEdit app**. This will provide Federal Reserve staff with a centralized interface for AI-powered research assistance.

### Integration Scope

**Frontend Consolidation**:
- Unified landing page with navigation to Research Agents, MarginalEdit, and future tools
- Single sign-on (SSO) authentication for all tools
- Consistent UI/UX design language across applications

**Backend Infrastructure** (handled by Board AI staff):
- **Hosting**: Determine production environment (on-prem servers, cloud deployment)
- **API Key Management**: Centralized credential storage and rotation
- **Database**: Persistent storage for evaluation history, user preferences, cached results
- **Monitoring**: Logging, alerting, performance metrics for all integrated tools
- **Scalability**: Load balancing, rate limiting, cost controls

### Division of Responsibilities

**Research Agents Team** (current maintainers):
- Maintain core evaluation logic (`app_system/referee/`, `section_eval/`)
- Develop and refine prompts
- Run benchmarking experiments
- Provide API/interface for integration
- Document system behavior and limitations

**Board AI Staff**:
- Software infrastructure (hosting, deployment, CI/CD)
- Authentication and authorization
- Database schema design and management
- API gateway and routing
- Cross-tool integrations
- Production monitoring and incident response
- Compliance and security (data handling, audit logs)

### Integration Timeline

**Phase 1** (Q2-Q3 2026): Standalone deployment
- Deploy Research Agents app as-is on Board infrastructure
- Establish hosting environment and API key management
- Basic monitoring and logging

**Phase 2** (Q3-Q4 2026): Unified frontend
- Build landing page with navigation to Research Agents + MarginalEdit
- Implement SSO authentication
- Share styling and common UI components

**Phase 3** (2027+): Deep integration
- Shared database for cross-tool features (e.g., MarginalEdit suggests revisions based on Research Agents critiques)
- Workflow automation (evaluate → revise → re-evaluate loop)
- Admin dashboard for usage analytics and cost monitoring

### Technical Considerations

**API Design**:
- Current app is monolithic Streamlit app
- May need to expose REST API for frontend/backend separation
- Consider: FastAPI wrapper around `execute_debate_pipeline()` and `SectionEvaluatorApp`

**Data Persistence**:
- Currently results are ephemeral (session state only)
- Future: Store evaluation history in database
- Schema considerations: users, papers, evaluations, debate transcripts, prompts used

**Configuration Management**:
- Currently `.env` file for API credentials
- Future: Centralized config service (e.g., AWS Secrets Manager, HashiCorp Vault)

**Scalability**:
- Current: Single-instance Streamlit app
- Future: Horizontal scaling for concurrent users (requires stateless design)

## 4. Research Directions

### Consistency Improvements

**Ongoing work** (see `running-ideas.md` for full roadmap):
- **Phase 1** (✅ Complete): Removed generic system prompt pollution
- **Phase 2** (✅ Complete): Per-round temperature control
- **Phase 3** (Planned): Structured output formats (JSON schema enforcement)
- **Phase 4** (Exploratory): Deterministic consensus algorithms

### Persona Expansion

**11th Persona Candidates**:
- **Replication Specialist**: Focus on reproducibility, data availability, code quality
- **Experimentalist**: Lab/field experiments, RCT design
- **Domain Expert**: Economics sub-field expertise (labor, trade, finance, etc.)

**Considerations**:
- Does adding personas improve quality or just increase cost?
- Round 0 selection already chooses 3 of 10—does expanding pool help?

### Alternative Architectures

**Variants to explore**:
1. **Sequential debate**: Personas respond in order rather than parallel (simulates real committee discussions)
2. **Adversarial debate**: Explicitly assign "advocate" vs. "critic" roles
3. **Bayesian aggregation**: Treat persona votes as noisy signals, use formal belief updating
4. **Recursive refinement**: Multi-iteration debate (currently only 1 pass through rounds)

### Cross-Domain Applications

**Beyond economics papers**:
- **Policy memos**: Already prototyped in `app-memo.py`, `memo_engine.py`
- **Grant proposals**: NSF/NIH proposal evaluation
- **Code review**: Software engineering peer review (requires code-specific personas)
- **Medical literature**: Clinical trial evaluation (requires domain expertise)

## 5. Risks & Mitigation

### Technical Risks

**API Dependency**:
- **Risk**: Claude API changes, rate limits, downtime
- **Mitigation**: Abstract LLM calls behind interface, support multiple providers (OpenAI, Gemini fallback)

**Cost Escalation**:
- **Risk**: High token usage for benchmarking experiment
- **Mitigation**: Monitor costs closely, use caching aggressively, consider cheaper models for non-critical rounds

**Model Drift**:
- **Risk**: Claude 4.5 behavior changes over time (Anthropic model updates)
- **Mitigation**: Version lock model IDs, test new versions before switching, maintain prompt version history

### Research Risks

**Benchmark Validity**:
- **Risk**: Ground truth tiers may not reflect paper quality (publication decisions are noisy)
- **Mitigation**: Treat as rough proxy, supplement with qualitative evaluation, consider multiple metrics

**Overfitting to Economics**:
- **Risk**: System is too specialized for Federal Reserve economics papers
- **Mitigation**: Test on external datasets (arXiv, SSRN), design for generalization

**Ethical Considerations**:
- **Risk**: Over-reliance on AI evaluation leads to homogenization (papers that "game" the system)
- **Mitigation**: Position as decision support tool, not replacement for human judgment

### Organizational Risks

**Transition/Handoff**:
- **Risk**: Knowledge loss as team members leave
- **Mitigation**: Maintain comprehensive documentation (`CLAUDE.md`, `handoff_context.md`), use Claude Code for continuity

**Resource Allocation**:
- **Risk**: Insufficient staff time for maintenance + new development
- **Mitigation**: Prioritize critical path (benchmarking experiment), defer nice-to-have features

## 6. Success Metrics

### Short-term (6 months)

- ✅ Complete benchmarking experiment on full ground truth dataset
- ✅ Achieve >70% accuracy on Tier 1 vs. Tier 4 classification
- ✅ Deploy stable production version on Board AI infrastructure
- ✅ Integrate with unified frontend landing page

### Medium-term (1 year)

- ✅ 50+ Federal Reserve staff have used the system for paper evaluation
- ✅ Positive user feedback (>4/5 rating on usefulness)
- ✅ Cost per evaluation <$2.00 (through caching and optimization)
- ✅ Systematic consistency improvements (Phase 3+ from `running-ideas.md`)

### Long-term (2+ years)

- ✅ Published research paper on MAD architecture and benchmarking results
- ✅ Cross-domain application (policy memos, grant proposals, etc.)
- ✅ Open-source release (pending legal/compliance review)
- ✅ Adoption by other Federal Reserve banks or external institutions

## 7. Getting Started

**For new maintainers**:

1. **Read documentation**:
   - `handoff_context.md` (this project)
   - `CLAUDE.md` (technical architecture)
   - `running-ideas.md` (ongoing research directions)

2. **Set up local environment**:
   ```bash
   cd app_system
   cp .env.example .env  # Add API credentials
   source ../venv/bin/activate
   streamlit run app.py
   ```

3. **Run tests**:
   ```bash
   cd app_system
   pytest tests/
   ```

4. **Review experiment setup**:
   ```bash
   cd experiment
   python test_setup.py  # Verify environment
   ```

5. **Contact Board AI staff** for:
   - Production deployment access
   - API key provisioning
   - Database credentials
   - Integration timeline

---

**Questions?** Refer to `CLAUDE.md` for technical details or reach out to Board AI program leadership for infrastructure/integration questions.
