
# --- ORCHESTRATION PIPELINE FOR DYNAMIC 'N' ---
async def run_round_0_selection(paper_file, N) -> dict:
    print(f"\n--- STARTING ROUND 0: ENDOGENOUS SELECTION (N={N}) ---")
    prompt = SELECTION_PROMPT.format(N=N)
    response = await call_llm_serial("You are an AI routing agent.", prompt, "Editor_Selector", files=[paper_file])
    
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        selection_data = json.loads(json_match.group())
        personas = selection_data.get("selected_personas", [])
        if len(personas) != N:
            print(f"[WARNING] Requested {N} personas, but LLM returned {len(personas)}.")
        print(f"[SYSTEM] Selected Panel: {personas}")
        return selection_data
    except Exception as e:
        print(f"[ERROR] JSON parse failed. Defaulting to first {N} options. Error: {e}")
        # Dynamic fallback based on N
        default_panel = list(SYSTEM_PROMPTS.keys())[:N]
        eq_weight = round(1.0 / N, 2)
        return {
            "selected_personas": default_panel,
            "weights": {p: eq_weight for p in default_panel}
        }

async def run_round_1_serial(active_personas, paper_file) -> dict:
    print("\n--- STARTING ROUND 1: INDEPENDENT EVALUATION ---")
    reports = {}
    for role in active_personas:
        reports[role] = await call_llm_serial(SYSTEM_PROMPTS[role], "Evaluate this paper based on your role.", role, files=[paper_file])
        await asyncio.sleep(20) # Gentle cooldown between personas
    return reports

async def run_round_2a_serial(r1_reports, active_personas, paper_file) -> dict:
    print("\n--- STARTING ROUND 2A: CROSS-EXAMINATION ---")
    reports = {}
    for role in active_personas:
        peers = [p for p in active_personas if p != role]
        peer_reports_text = "\n".join([f"--- {p} Report ---\n{r1_reports[p]}\n" for p in peers])
        
        prompt_2a = DEBATE_PROMPTS["Round_2A_Cross_Examination"].format(
            role=role, peer_reports=peer_reports_text
        )
        reports[role] = await call_llm_serial(SYSTEM_PROMPTS[role], prompt_2a, role, files=[paper_file])
        await asyncio.sleep(20)
    return reports

async def run_round_2b_serial(r2a_reports, active_personas, paper_file) -> dict:
    print("\n--- STARTING ROUND 2B: ANSWERING CLARIFICATIONS ---")
    r2a_transcript = "\n".join([f"[{r} CROSS-EXAMINATION]:\n{text}\n" for r, text in r2a_reports.items()])
    
    reports = {}
    for role in active_personas:
        prompt_2b = DEBATE_PROMPTS["Round_2B_Direct_Examination"].format(
            role=role, r2a_transcript=r2a_transcript
        )
        reports[role] = await call_llm_serial(SYSTEM_PROMPTS[role], prompt_2b, role, files=[paper_file])
        await asyncio.sleep(20)
    return reports

async def run_round_2c_serial(r1_reports, r2a_reports, r2b_reports, active_personas, paper_file) -> dict:
    print("\n--- STARTING ROUND 2C: FINAL AMENDMENTS ---")
    transcript = "ROUND 1 REPORTS:\n" + "\n".join([f"[{r}]:\n{t}" for r, t in r1_reports.items()])
    transcript += "\nCROSS-EXAMINATION:\n" + "\n".join([f"[{r}]:\n{t}" for r, t in r2a_reports.items()])
    transcript += "\nANSWERS:\n" + "\n".join([f"[{r}]:\n{t}" for r, t in r2b_reports.items()])

    reports = {}
    for role in active_personas:
        prompt_2c = DEBATE_PROMPTS["Round_2C_Final_Amendment"].format(role=role, debate_transcript=transcript)
        reports[role] = await call_llm_serial(SYSTEM_PROMPTS[role], prompt_2c, role, files=[paper_file])
        await asyncio.sleep(20)
    return reports

async def run_round_3(r2c_reports, selection_data) -> str:
    print("\n--- STARTING ROUND 3: EDITOR DECISION ---")
    weights_json = json.dumps(selection_data["weights"], indent=2)
    final_reports_text = "\n".join([f"--- {r.upper()} FINAL REPORT ---\n{t}" for r, t in r2c_reports.items()])
    
    prompt_3 = DEBATE_PROMPTS["Round_3_Editor"].format(weights_json=weights_json, final_reports_text=final_reports_text)
    return await call_llm_serial("You are the Senior Editor.", prompt_3, "Editor")

async def execute_experiment(paper_file, N_personas):
    """Wraps the pipeline for Phase 4 experiments."""
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

    # Pipeline execution
    selection_data = await run_round_0_selection(paper_file, N=N_personas)
    active_personas = selection_data["selected_personas"]
    log_stage(f"ROUND 0: SELECTION (N={N_personas})", single_text=json.dumps(selection_data, indent=2))
    
    r1_reports = await run_round_1_serial(active_personas, paper_file)
    log_stage("ROUND 1: INDEPENDENT EVALUATION", reports=r1_reports)
    
    r2a_reports = await run_round_2a_serial(r1_reports, active_personas, paper_file)
    log_stage("ROUND 2A: CROSS-EXAMINATION", reports=r2a_reports)
    
    r2b_reports = await run_round_2b_serial(r2a_reports, active_personas, paper_file)
    log_stage("ROUND 2B: ANSWERING CLARIFICATIONS", reports=r2b_reports)
    
    r2c_reports = await run_round_2c_serial(r1_reports, r2a_reports, r2b_reports, active_personas, paper_file)
    log_stage("ROUND 2C: FINAL AMENDMENTS", reports=r2c_reports)
    
    final_decision = await run_round_3(r2c_reports, selection_data)
    log_stage("FINAL REFEREE REPORT (EDITOR)", single_text=final_decision)

    print(f"\n[SUCCESS] Experiment N={N_personas} completed!")
    return final_decision

#Sample execution 
# Set the number of personas you want the LLM to endogenously select
TARGET_N = 3

# Ensure paper1 is defined from setup file
print(f"Executing pipeline with {TARGET_N} dynamically selected personas...")

# Initialize the orchestrator
final_output = await execute_experiment(paper1, N_personas=TARGET_N)

print("\n" + "="*60)
print(f"FINAL OUTPUT FOR N={TARGET_N}")
print("="*60)
print(final_output)
