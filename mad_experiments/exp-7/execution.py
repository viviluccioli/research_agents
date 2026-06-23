# ==========================================
# ORCHESTRATION FUNCTIONS
# ==========================================

async def run_round_0_selection(paper_file, N=3):
    """Parses JSON to dynamically select personas and weights."""
    prompt = SELECTION_PROMPT.format(N=N)

    raw_response = await call_llm_serial("You are the Chief Editor.", prompt, "Editor", files=[paper_file], temperature=0.0)

    # Structural fix: Guard against None AND empty strings
    if not raw_response or not raw_response.strip():
        raise ValueError("Editor persona selection returned empty or None content. Check API status.")

    try:
        # Strip markdown format tags cleanly
        clean_json = raw_response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        return data
    except json.JSONDecodeError as e:
        print(f"[Error] Failed to parse Editor selection JSON. Raw response:\n{raw_response}")
        raise e


async def run_round_1_serial(active_personas, paper_file, temperature=0.2):
    """Executes the independent evaluation round."""
    reports = {}
    for role in active_personas:
        prompt = f"Read the provided paper and execute your Domain Audit based on your System Instructions."
        # Use the passed temperature variable here
        reports[role] = await call_llm_serial(SYSTEM_PROMPTS[role], prompt, role, files=[paper_file], temperature=temperature)
        await asyncio.sleep(15) # Rate limit pacing
    return reports

async def execute_experiment(paper_file, N_personas, decision_mode="probabilistic_dung"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = os.path.join(OUTPUT_DIR, f"Experiment_N{N_personas}_{decision_mode}")
    os.makedirs(exp_folder, exist_ok=True)

    log_file_path = os.path.join(exp_folder, f"Review_{timestamp}.txt")
    print(f"\n[SYSTEM] Executing mode '{decision_mode}'. Outputting to: {log_file_path}")

    def log_stage(stage_title, content_string):
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n{stage_title}\n{'='*60}\n")
            f.write(content_string + "\n")

    # --- ROUND 0: SELECTION ---
    selection_data = await run_round_0_selection(paper_file, N=N_personas)
    active_personas = selection_data.get("selected_personas", list(SYSTEM_PROMPTS.keys())[:N_personas])
    weights_dict = selection_data.get("weights", {p: 1.0/N_personas for p in active_personas})

    # Log ONLY Selection & Justification
    selection_log = f"Personas: {', '.join(active_personas)}\nJustification: {selection_data.get('justification', '')}"
    log_stage(f"ROUND 0: SELECTION", selection_log)

    # --- ROUND 1: INDEPENDENT EVALUATION ---
    print("\n--- RUNNING ROUND 1: INDEPENDENT EVALUATION (Hidden from console) ---")
    r1_reports = await run_round_1_serial(active_personas, paper_file, temperature=0.0)

    # Log ONLY Domain Audit and Severity
    r1_log = ""
    for role, text in r1_reports.items():
        r1_log += f"\n--- {role.upper()} ---\n{extract_round_1_essentials(text)}\n"
    log_stage("ROUND 1: DOMAIN AUDITS & SEVERITY", r1_log)

    # --- HITL ---
    print("\n" + "!"*40 + "\n🛑 HITL: ROUND 1 COMPLETE 🛑\n" + "!"*40)
    human_input = input("Enter Lead Author Directives (or Enter to proceed): ")
    human_directive = f"LEAD AUTHOR DIRECTIVE: {human_input}" if human_input.strip() else "No specific directive provided."

    # --- DYNAMIC DEBATE LOOP ---
    print("\n--- STARTING DYNAMIC LOGIC DEBATE ---")
    max_loops = 3
    current_loop = 1
    debate_history = r1_reports
    final_debate_state = r1_reports

    while current_loop <= max_loops:
        print(f"[SYSTEM] Executing Debate Loop {current_loop}...")
        compressed_context = compress_transcript(debate_history) # Uses full tags for LLM context
        current_reports = {}

        for role in active_personas:
            prompt_loop = DEBATE_PROMPTS["Dynamic_Debate_Round"].format(
                role=role, human_directive=human_directive, compressed_context=compressed_context
            )
            current_reports[role] = await call_llm_serial(SYSTEM_PROMPTS[role], prompt_loop, role, files=[paper_file], temperature=0.5)
            await asyncio.sleep(20)

        # Log ONLY the Dung Graph tags and Final Argument State
        loop_log = ""
        for role, text in current_reports.items():
            loop_log += f"\n--- {role.upper()} ---\n{extract_debate_essentials(text)}\n"
        log_stage(f"DEBATE LOOP {current_loop} (DUNG GRAPH)", loop_log)

        final_debate_state = current_reports

        # --- NEW EARLY EXIT BASED ON VERDICT CONSENSUS ---
        # Extract the standard verdict string ("PASS", "REVISE", "FAIL") for each persona
        current_verdicts = {role: extract_verdict(text) for role, text in current_reports.items()}

        # Get a set of unique verdicts present in this round
        unique_verdicts = set(current_verdicts.values())

        # If all active personas converge on the EXACT SAME verdict string, exit immediately
        if len(unique_verdicts) == 1:
            winning_verdict = list(unique_verdicts)[0]
            print(f"\n[SYSTEM] Early Stopping Triggered: All personas reached consensus on '{winning_verdict}'.")
            break

        debate_history = current_reports
        current_loop += 1

    # --- LAST ROUND: EDITOR DECISION---
    print(f"\n--- STARTING LAST ROUND: EDITOR DECISION ({decision_mode.upper()}) ---")

    # 1. Compute ALL three architectural scores for tracking and synthesis transparency
    vote_score = calculate_final_score(final_debate_state, weights_dict, mode="voting")
    dung_score = calculate_final_score(final_debate_state, weights_dict, mode="probabilistic_dung")
    weighted_score = calculate_final_score(final_debate_state, weights_dict, mode="weighted")

    # Set the actual functional score and mandate tracking based on your user argument
    if decision_mode == "voting":
        final_score = vote_score
    elif decision_mode == "probabilistic_dung":
        final_score = dung_score
    else:
        final_score = weighted_score

    if final_score > 0.75: mandated_decision = "ACCEPT"
    elif final_score >= 0.40: mandated_decision = "REJECT AND RESUBMIT"
    else: mandated_decision = "REJECT"

    print(f"[SYSTEM] Score ({decision_mode}): {final_score:.2f}/1.0 -> {mandated_decision}")

    # 2. Build the explicit Calculation Audit String
    audit_string = f"""
============================================================
MECHANICAL CALCULATION AUDIT (TRIPLE METRIC OVERLAY)
============================================================
* Active Persona Weights: {json.dumps(weights_dict, indent=2)}
* Evaluation Run ID: Review_20260617_194449.txt

* Metric Breakdown:
  - Mode [VOTING]:              {vote_score:.2f} / 1.0
  - Mode [PROBABILISTIC_DUNG]:  {dung_score:.2f} / 1.0 (Selected Mode)
  - Mode [WEIGHTED_AVERAGE]:     {weighted_score:.2f} / 1.0
------------------------------------------------------------
MANDATED SYSTEM RESOLUTION: {mandated_decision} (Score: {final_score:.2f})
============================================================
"""

    # 3. Inject the audit string directly into the LLM synthesis template
    # Make sure your DEBATE_PROMPTS["Final_Round_Editor"] string contains the {calculation_audit} token!
    prompt_3 = DEBATE_PROMPTS["Final_Round_Editor"].format(
        python_calculated_score=round(final_score, 2),
        python_mandated_decision=mandated_decision,
        calculation_audit=audit_string,
        final_compressed_transcript=compress_transcript(final_debate_state)
    )

    final_decision_letter = await call_llm_serial("You are the Senior Editor.", prompt_3, "Editor", temperature=0.2)

    # 4. Prepend the audit string directly to the final file output log to fix clunky formatting
    comprehensive_output = f"{audit_string}\n\n{final_decision_letter}"

    log_stage("ROUND 3: FINAL EDITOR SYNTHESIS & AUDIT", comprehensive_output)

    return comprehensive_output

#to run: await execute_experiment(paper_file=NAME HERE, N_personas=NUMBER OF DESIRED PERSONAS)
