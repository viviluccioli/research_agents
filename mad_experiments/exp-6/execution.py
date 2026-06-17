# ==========================================
# ORCHESTRATION FUNCTIONS
# ==========================================

async def run_round_0_selection(paper_file, N=3):
    """Parses JSON to dynamically select personas and weights."""
    prompt = SELECTION_PROMPT.format(N=N)
    
    raw_response = await call_llm_serial("You are the Chief Editor.", prompt, "Editor", files=[paper_file], temperature=0.0)
    
    # Check if raw_response is None
    if raw_response is None:
        raise ValueError("Editor persona selection returned None. Check API status or model availability.")
        
    try:
        # Now it is safe to call replace
        clean_json = raw_response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        return data
    except json.JSONDecodeError as e:
        print(f"[Error] Failed to parse Editor selection JSON. Raw response:\n{raw_response}")
        raise e

async def run_round_1_serial(active_personas, paper_file):
    """Executes the independent evaluation round."""
    reports = {}
    for role in active_personas:
        prompt = f"Read the provided paper and execute your Domain Audit based on your System Instructions."
        reports[role] = await call_llm_serial(SYSTEM_PROMPTS[role], prompt, role, files=[paper_file], temperature= 0.2)
        await asyncio.sleep(15) # Rate limit pacing
    return reports

async def execute_experiment(paper_file, N_personas):
    """Wraps the pipeline for Phase 4 experiments with Dynamic Consensus."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = os.path.join(OUTPUT_DIR, f"Experiment_N{N_personas}")
    os.makedirs(exp_folder, exist_ok=True)

    log_file_path = os.path.join(exp_folder, f"Review_N{N_personas}_{timestamp}.txt")
    print(f"\n[SYSTEM] Saving outputs to: {log_file_path}")

    def log_stage(stage_title, reports=None, single_text=None):
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n{stage_title}\n{'='*60}\n")
            if reports:
                for role, content in reports.items():
                    f.write(f"\n--- {role.upper()} ---\n{content}\n")
            elif single_text:
                f.write(single_text + "\n")

    # ------------------------------------------
    # ROUND 0 & 1: SELECTION & INDEPENDENT EVAL
    # ------------------------------------------
    selection_data = await run_round_0_selection(paper_file, N=N_personas)
    active_personas = selection_data.get("selected_personas", list(SYSTEM_PROMPTS.keys())[:N_personas])
    weights_dict = selection_data.get("weights", {p: 1.0/N_personas for p in active_personas})

    log_stage(f"ROUND 0: SELECTION (N={N_personas})", single_text=json.dumps(selection_data, indent=2))

    print("\n--- STARTING ROUND 1: INDEPENDENT EVALUATION ---")
    r1_reports = await run_round_1_serial(active_personas, paper_file)
    log_stage("ROUND 1: INDEPENDENT EVALUATION", reports=r1_reports)

    # ------------------------------------------
    # HUMAN-IN-THE-LOOP PAUSE
    # ------------------------------------------
    print("\n" + "!"*40 + "\n🛑 HITL: REVIEW ROUND 1 ABOVE 🛑\n" + "!"*40)
    human_input = input("Enter Lead Author Directives (or Enter to proceed): ")
    human_directive = f"LEAD AUTHOR DIRECTIVE: {human_input}" if human_input.strip() else "No specific directive provided."
    log_stage("LEAD AUTHOR DIRECTIVE", single_text=human_directive)

    # ------------------------------------------
    # DYNAMIC DEBATE LOOP
    # ------------------------------------------
    print("\n--- STARTING DYNAMIC LOGIC DEBATE ---")
    max_loops = 3
    current_loop = 1
    debate_history = r1_reports
    final_debate_state = r1_reports # Fallback

    while current_loop <= max_loops:
        print(f"\n[SYSTEM] Executing Debate Loop {current_loop} / {max_loops}...")
        compressed_context = compress_transcript(debate_history)
        current_reports = {}

        for role in active_personas:
            prompt_loop = DEBATE_PROMPTS["Dynamic_Debate_Round"].format(
                role=role,
                human_directive=human_directive,
                compressed_context=compressed_context
            )
            current_reports[role] = await call_llm_serial(SYSTEM_PROMPTS[role], prompt_loop, role, files=[paper_file], temperature=0.5)
            await asyncio.sleep(20) # Gentle cooldown between personas

        log_stage(f"DEBATE LOOP {current_loop}", reports=current_reports)
        final_debate_state = current_reports

        # --- CONSENSUS CHECK (Early Stop) ---
        verdicts = [extract_verdict(text) for text in current_reports.values()]

        if all(v == "PASS" for v in verdicts):
            print("\n[SYSTEM] Unanimous PASS achieved. Stopping debate early.")
            break
        elif all(v == "FAIL" for v in verdicts):
            print("\n[SYSTEM] Unanimous FAIL achieved. Stopping debate early.")
            break
        elif not any("[ATTACK" in text for text in current_reports.values()) and current_loop > 1:
            print("\n[SYSTEM] All attacks resolved or conceded. Stopping debate early.")
            break

        debate_history = current_reports
        current_loop += 1

    # ------------------------------------------
    # ROUND 3: PYTHON ARITHMETIC & EDITOR SYNTHESIS
    # ------------------------------------------
    print("\n--- STARTING ROUND 3: EDITOR DECISION ---")

    # 1. Execute mathematical extraction
    final_score = calculate_final_score(final_debate_state, weights_dict, mode="weighted")

    # Threshold Logic
    if final_score > 0.75:
        mandated_decision = "ACCEPT"
    elif final_score >= 0.40:
        mandated_decision = "REVISE AND RESUBMIT"
    else:
        mandated_decision = "REJECT"

    print(f"[SYSTEM] Python Score Calculated: {final_score}/1.0 -> {mandated_decision}")

    # 2. Feed the result back to the Editor for letter synthesis
    final_compressed_transcript = compress_transcript(final_debate_state)
    prompt_3 = DEBATE_PROMPTS["Round_3_Editor"].format(
        python_calculated_score=final_score,
        python_mandated_decision=mandated_decision,
        final_compressed_transcript=final_compressed_transcript
    )

    final_decision_letter = await call_llm_serial("You are the Senior Editor.", prompt_3, "Editor")
    log_stage("FINAL REFEREE REPORT (EDITOR)", single_text=final_decision_letter)

    print(f"\n[SUCCESS] Experiment N={N_personas} completed! Output saved to: {log_file_path}")
    return final_decision_letter


#To run 
