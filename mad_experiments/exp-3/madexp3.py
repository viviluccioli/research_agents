import time
import os
import asyncio
import datetime
import json
import re
from google import genai
from google.genai import errors, types
from google.colab import drive, userdata
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
drive.mount('/content/drive')
API_KEY = userdata.get('GEMINI_API_KEY')
client = genai.Client(api_key=API_KEY)

# Ensure the output directory exists in your Drive
OUTPUT_DIR = "/content/drive/MyDrive/SENIOR_YEAR/FED"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@retry(
    stop=stop_after_attempt(12),
    wait=wait_exponential(multiplier=10, min=30, max=300),
    retry=retry_if_exception_type((errors.ClientError, Exception))
)
def generate_safe_content(system_prompt: str, user_prompt: str, files=None):
    """Synchronous Gemini call with extreme backoff for Free Tier survival."""
    try:
        content_list = [user_prompt]
        if files:
            content_list.extend(files)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.5
            )
        )
        return response.text
    except Exception as e:
        msg = str(e).upper()
        if "429" in msg or "QUOTA" in msg:
            print("  [API] Rate limit (429) reached. Waiting 30s+ before retry...")
            raise e
        elif "503" in msg or "UNAVAILABLE" in msg:
            print("  [API] Server overloaded (503). Retrying...")
            raise e
        else:
            raise e

async def call_llm_serial(system_prompt: str, user_prompt: str, role: str, files=None) -> str:
    """Async wrapper. Delays are now handled explicitly in the orchestration loops."""
    print(f"[{role}] Generating response...")
    return await asyncio.to_thread(generate_safe_content, system_prompt, user_prompt, files)

def wait_for_files_active(files):
    """Waits for a list of files to be processed and 'ACTIVE'."""
    print("Verifying file status...")
    for f in files:
        file_status = client.files.get(name=f.name)
        while file_status.state.name == "PROCESSING":
            print(f"  ...{f.display_name} is still processing. Waiting 2s.")
            time.sleep(2)
            file_status = client.files.get(name=f.name)

        if file_status.state.name == "FAILED":
            raise Exception(f"File {f.display_name} failed to process.")

        print(f"  ✓ {f.display_name} is ACTIVE and ready.")

# ==========================================
# 2. FILE UPLOAD; change according to local file location
# ==========================================
paper1 = client.files.upload(file='/content/drive/MyDrive/SENIOR_YEAR/FED/MadTest1.pdf')
wait_for_files_active([paper1])
print(f"Uploaded {paper1.display_name} as {paper1.name}")

# ==========================================
# 3. PAPER TYPE CONTEXT (optional)
# Set paper_type to "empirical", "theoretical", "policy", or None to skip.
# ==========================================
paper_type = None  # <-- set this before running

# Prompt files are loaded from the app_system prompts directory.
# Update PROMPTS_DIR if your directory structure differs.
PROMPTS_DIR = "/content/drive/MyDrive/SENIOR_YEAR/FED/app_system/prompts/multi_agent_debate"

def load_prompt_file(path: str) -> str:
    """Load a prompt .txt file, returning empty string on failure."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"  [WARN] Could not load prompt file {path}: {e}")
        return ""

def load_paper_type_context(paper_type: str) -> str:
    if not paper_type:
        return ""
    path = f"{PROMPTS_DIR}/additional_context/paper_type_contexts/{paper_type}/v1.0.txt"
    return load_prompt_file(path)

_ERROR_SEVERITY_GUIDE = load_prompt_file(
    f"{PROMPTS_DIR}/additional_context/error_severity/v1.0.txt"
)

# ==========================================
# 4. PROMPTS
# ==========================================

SELECTION_PROMPT = load_prompt_file(f"{PROMPTS_DIR}/debate_rounds/round_0_selection/v1.0.txt")

# Persona system prompts — loaded from versioned .txt files.
# The {error_severity} placeholder in each file is replaced with the loaded severity guide.
def _load_persona_prompts() -> dict:
    personas = ["Theorist", "Empiricist", "Historian", "Visionary", "Policymaker"]
    prompts = {}
    for name in personas:
        raw = load_prompt_file(f"{PROMPTS_DIR}/personas/{name.lower()}/v1.0.txt")
        prompts[name] = raw.replace("{error_severity}", _ERROR_SEVERITY_GUIDE)
    return prompts

SYSTEM_PROMPTS = _load_persona_prompts()

# Debate round prompts — loaded from versioned .txt files.
_r2a = load_prompt_file(f"{PROMPTS_DIR}/debate_rounds/round_2a_cross_exam/v1.0.txt")
_r2b = load_prompt_file(f"{PROMPTS_DIR}/debate_rounds/round_2b_direct_exam/v1.0.txt")
_r2c = load_prompt_file(f"{PROMPTS_DIR}/debate_rounds/round_2c_final_amendment/v1.0.txt")
_r3  = load_prompt_file(f"{PROMPTS_DIR}/debate_rounds/round_3_editor/v1.0.txt")

DEBATE_PROMPTS = {
    "Round_2A_Cross_Examination": _r2a,
    "Round_2B_Direct_Examination": _r2b,
    "Round_2C_Final_Amendment":    _r2c,
    "Round_3_Editor":              _r3,
}

# ==========================================
# 5. ORCHESTRATION PIPELINE
# ==========================================

async def run_round_0_selection(paper_file, paper_type=None) -> dict:
    """
    Dynamically selects the 3 most relevant personas and their weights.
    Optionally incorporates paper type context guidance.
    """
    print("\n--- STARTING ROUND 0: ENDOGENOUS PERSONA SELECTION ---")

    # Build the selection prompt, injecting paper type context if provided
    full_selection_prompt = SELECTION_PROMPT

    if paper_type and paper_type in PAPER_TYPE_CONTEXTS:
        print(f"  [SYSTEM] Injecting paper type context: {paper_type}")
        full_selection_prompt += f"\n\n{PAPER_TYPE_CONTEXTS[paper_type]}\n"

    full_selection_prompt += "\nPAPER TEXT: [See attached file]"

    response = await call_llm_serial(
        system_prompt="You are an AI routing agent. Follow instructions STRICTLY.",
        user_prompt=full_selection_prompt,
        role="Editor_Selector",
        files=[paper_file]
    )

    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        selection_data = json.loads(json_match.group())

        personas = selection_data.get("selected_personas", [])
        weights = selection_data.get("weights", {})
        if len(personas) != 3:
            raise ValueError("LLM did not select exactly 3 personas.")

        print(f"[SYSTEM] Endogenous Selection Complete. Panel: {personas}")
        print(f"[SYSTEM] Weights: {weights}")
        print(f"[SYSTEM] Justification: {selection_data.get('justification', '')}")
        return selection_data

    except Exception as e:
        print(f"[ERROR] Failed to parse Selection JSON. Defaulting to Empiricist, Historian, Policymaker. Error: {e}")
        return {
            "selected_personas": ["Empiricist", "Historian", "Policymaker"],
            "weights": {"Empiricist": 0.4, "Historian": 0.3, "Policymaker": 0.3},
            "justification": "Default selection due to parsing error."
        }

async def run_round_1_serial(active_personas: list, paper_file) -> dict:
    print("\n--- STARTING ROUND 1: INDEPENDENT EVALUATION (SERIAL) ---")
    user_prompt = "Please read the attached paper and provide your Round 1 evaluation based on your role."
    reports = {}

    for role in active_personas:
        reports[role] = await call_llm_serial(SYSTEM_PROMPTS[role], user_prompt, role, files=[paper_file])
        print(f"  [SYSTEM] {role} finished. Cooling down for 30s...")
        await asyncio.sleep(30)

    return reports

async def run_round_2a_serial(r1_reports: dict, active_personas: list, paper_file) -> dict:
    print("\n--- STARTING ROUND 2A: CROSS-EXAMINATION (SERIAL) ---")
    reports = {}

    for role in active_personas:
        peers = [p for p in active_personas if p != role]
        prompt_2a = DEBATE_PROMPTS["Round_2A_Cross_Examination"].format(
            role=role,
            peer_1_role=peers[0], peer_1_report=r1_reports[peers[0]],
            peer_2_role=peers[1], peer_2_report=r1_reports[peers[1]]
        )
        reports[role] = await call_llm_serial(SYSTEM_PROMPTS[role], prompt_2a, role, files=[paper_file])
        print(f"  [SYSTEM] {role} finished. Cooling down for 30s...")
        await asyncio.sleep(30)

    return reports

async def run_round_2b_serial(r2a_reports: dict, active_personas: list, paper_file) -> dict:
    print("\n--- STARTING ROUND 2B: ANSWERING CLARIFICATIONS (SERIAL) ---")
    r2a_transcript = ""
    for role, report in r2a_reports.items():
        r2a_transcript += f"\n[{role} CROSS-EXAMINATION]:\n{report}\n"

    reports = {}
    for role in active_personas:
        peers = [p for p in active_personas if p != role]
        prompt_2b = DEBATE_PROMPTS["Round_2B_Direct_Examination"].format(
            role=role,
            peer_1_role=peers[0],
            peer_2_role=peers[1],
            r2a_transcript=r2a_transcript
        )
        reports[role] = await call_llm_serial(SYSTEM_PROMPTS[role], prompt_2b, role, files=[paper_file])
        print(f"  [SYSTEM] {role} finished. Cooling down for 30s...")
        await asyncio.sleep(30)

    return reports

async def run_round_2c_serial(r1_reports: dict, r2a_reports: dict, r2b_reports: dict, active_personas: list, paper_file) -> dict:
    print("\n--- STARTING ROUND 2C: FINAL AMENDMENTS (SERIAL) ---")

    transcript = "ROUND 1 REPORTS:\n"
    for role, report in r1_reports.items():
        transcript += f"\n[{role}]:\n{report}\n"

    transcript += "\nCROSS-EXAMINATION (R2A):\n"
    for role, report in r2a_reports.items():
        transcript += f"\n[{role}]:\n{report}\n"

    transcript += "\nANSWERS & CONCESSIONS (R2B):\n"
    for role, report in r2b_reports.items():
        transcript += f"\n[{role}]:\n{report}\n"

    reports = {}
    for role in active_personas:
        prompt_2c = DEBATE_PROMPTS["Round_2C_Final_Amendment"].format(
            role=role, debate_transcript=transcript
        )
        reports[role] = await call_llm_serial(SYSTEM_PROMPTS[role], prompt_2c, role, files=[paper_file])
        print(f"  [SYSTEM] {role} finished. Cooling down for 30s...")
        await asyncio.sleep(30)

    return reports

async def run_round_3(r2c_reports: dict, selection_data: dict) -> str:
    print("\n--- STARTING ROUND 3: EDITOR DECISION ---")

    weights_json = json.dumps(selection_data["weights"], indent=2)

    final_reports_text = ""
    for role, report in r2c_reports.items():
        final_reports_text += f"\n--- {role.upper()} FINAL REPORT ---\n{report}\n"

    prompt_3 = DEBATE_PROMPTS["Round_3_Editor"].format(
        weights_json=weights_json,
        final_reports_text=final_reports_text
    )

    return await call_llm_serial(
        system_prompt="You are the Senior Editor. Follow the mathematical weighting instructions strictly.",
        user_prompt=prompt_3,
        role="Editor"
    )

async def execute_debate_pipeline(paper_file, paper_type=None):
    """Orchestrates the full MAD workflow, saves to Drive, and prints to console."""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = paper_file.display_name.replace('.pdf', '') if paper_file.display_name else "Unknown_Paper"
    log_file_path = os.path.join(OUTPUT_DIR, f"Review_{safe_name}_{timestamp}.txt")

    print(f"\n[SYSTEM] Saving live outputs to: {log_file_path}\n")

    def log_stage(stage_title, reports=None, single_text=None):
        with open(log_file_path, "a", encoding="utf-8") as f:
            header = f"\n{'='*60}\n{stage_title}\n{'='*60}\n"
            print(header)
            f.write(header)

            if reports:
                for role, content in reports.items():
                    body = f"\n--- {role.upper()} ---\n{content}\n"
                    print(body)
                    f.write(body)
            elif single_text:
                print(single_text)
                f.write(single_text + "\n")

    # Round 0: Selection (with optional paper type context)
    selection_data = await run_round_0_selection(paper_file, paper_type=paper_type)
    active_personas = selection_data["selected_personas"]
    log_stage("ROUND 0: PERSONA SELECTION", single_text=json.dumps(selection_data, indent=2))

    print("\n[SYSTEM] Inter-round cooldown for 30s...")
    await asyncio.sleep(30)

    # Round 1: Independent Evaluation
    r1_reports = await run_round_1_serial(active_personas, paper_file)
    log_stage("ROUND 1: INDEPENDENT EVALUATION", reports=r1_reports)

    print("\n[SYSTEM] Inter-round cooldown for 30s...")
    await asyncio.sleep(30)

    # Round 2A: Cross-Examination
    r2a_reports = await run_round_2a_serial(r1_reports, active_personas, paper_file)
    log_stage("ROUND 2A: CROSS-EXAMINATION", reports=r2a_reports)

    print("\n[SYSTEM] Inter-round cooldown for 30s...")
    await asyncio.sleep(30)

    # Round 2B: Answering Clarifications
    r2b_reports = await run_round_2b_serial(r2a_reports, active_personas, paper_file)
    log_stage("ROUND 2B: ANSWERING CLARIFICATIONS", reports=r2b_reports)

    print("\n[SYSTEM] Inter-round cooldown for 30s...")
    await asyncio.sleep(30)

    # Round 2C: Final Amendments
    r2c_reports = await run_round_2c_serial(r1_reports, r2a_reports, r2b_reports, active_personas, paper_file)
    log_stage("ROUND 2C: FINAL AMENDMENTS", reports=r2c_reports)

    print("\n[SYSTEM] Inter-round cooldown for 30s...")
    await asyncio.sleep(30)

    # Round 3: Final Synthesis
    final_decision = await run_round_3(r2c_reports, selection_data)
    log_stage("FINAL REFEREE REPORT (EDITOR)", single_text=final_decision)

    print(f"\n[SYSTEM] Pipeline fully executed! Transcript safely saved to your Google Drive at {OUTPUT_DIR}")
    return final_decision

# ==========================================
# 6. EXECUTION
# ==========================================
# Set paper_type above (line ~70), then run:
await execute_debate_pipeline(paper1, paper_type=paper_type)
