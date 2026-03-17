import time
import os
import asyncio
import datetime
import json
import re
import base64
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
API_KEY = "sk-rjg7EvJ1zJN35I5I4Jo1dg"
API_BASE = "https://martinai-preview-api.frb.gov/v1"
url_chat_completions = f"{API_BASE}/chat/completions"
model_selection = "anthropic.claude-sonnet-4-5-20250929-v1:0"

# Configure your local paths
OUTPUT_DIR = "./"  # Change this to your desired output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. FILE HANDLING
# ==========================================
def encode_pdf_to_base64(file_path: str) -> str:
    """Encode a PDF file to base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def create_file_content(file_path: str) -> dict:
    """Create file content dict for Claude API."""
    pdf_data = encode_pdf_to_base64(file_path)
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": pdf_data
        }
    }

class LocalFile:
    """Simple wrapper to mimic the Gemini file object."""
    def __init__(self, path: str):
        self.path = path
        self.display_name = os.path.basename(path)
        self.name = path

# ==========================================
# 3. THE "RESILIENT" LLM CALL (Serial Edition)
# ==========================================
@retry(
    stop=stop_after_attempt(12),
    wait=wait_exponential(multiplier=10, min=30, max=300),
    retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
)
def generate_safe_content(system_prompt: str, user_prompt: str, files=None):
    """Synchronous Claude API call with extreme backoff."""
    try:
        # Build the messages array
        messages = []
        
        # If files are provided, create a multimodal message
        if files:
            content_parts = []
            
            # Add file content first
            for file_obj in files:
                file_content = create_file_content(file_obj.path)
                content_parts.append(file_content)
            
            # Add text prompt
            content_parts.append({
                "type": "text",
                "text": user_prompt
            })
            
            messages.append({
                "role": "user",
                "content": content_parts
            })
        else:
            # Text-only message
            messages.append({
                "role": "user",
                "content": user_prompt
            })
        
        # Make the API call
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        payload = {
            "model": model_selection,
            "messages": messages,
            "system": system_prompt,
            "temperature": 0.5,
            "max_tokens": 4096
        }
        
        response = requests.post(
            url_chat_completions,
            headers=headers,
            json=payload,
            timeout=300
        )
        
        # Check for errors
        if response.status_code == 429:
            print("  [API] Rate limit (429) reached. Waiting before retry...")
            raise requests.exceptions.RequestException("Rate limit exceeded")
        elif response.status_code == 503:
            print("  [API] Server overloaded (503). Retrying...")
            raise requests.exceptions.RequestException("Service unavailable")
        elif response.status_code != 200:
            print(f"  [API] Error {response.status_code}: {response.text}")
            raise requests.exceptions.RequestException(f"API error: {response.status_code}")
        
        # Parse response
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.Timeout:
        print("  [API] Request timeout. Retrying...")
        raise
    except Exception as e:
        print(f"  [API] Error: {str(e)}")
        raise

async def call_llm_serial(system_prompt: str, user_prompt: str, role: str, files=None) -> str:
    """Async wrapper. Delays are handled explicitly in the orchestration loops."""
    print(f"[{role}] Generating response...")
    return await asyncio.to_thread(generate_safe_content, system_prompt, user_prompt, files)

# ==========================================
# 4. PROMPTS (UNCHANGED)
# ==========================================

SELECTION_PROMPT = """
You are the Chief Editor of an economics journal. You must select exactly THREE expert personas to review the provided paper.
The available personas are:
1. "Theorist": Focuses on formal mathematical proofs, logic, and model insight.
2. "Empiricist": Focuses on data, econometrics, identification strategy, and statistical validity.
3. "Historian": Focuses on literature lineage, historical background, and appropriate situating of the paper in relevant context.
4. "Visionary": Focuses on novelty and intellectual impact.
5. "Policymaker": Focuses on real-world application, welfare implications, and policy relevance.

Select the 3 most crucial personas for reviewing this specific paper. Assign them weights based on their relative importance to assessing THIS SPECIFIC PAPER. The weights must sum exactly to 1.0.

OUTPUT FORMAT: Return ONLY a valid JSON object. No markdown formatting, no explanations.
{
  "selected_personas": ["Persona1", "Persona2", "Persona3"],
  "weights": {
    "Persona1": 0.4,
    "Persona2": 0.35,
    "Persona3": 0.25
  },
  "justification": "1 sentence explaining the choice and weights."
}
"""

SYSTEM_PROMPTS = {
    "Theorist": """
    ### ROLE
    You are a rigorous Economic Theorist. You focus on mathematical logic, proofs, correct derivations, and models with mathematical insight. You value other perspectives—understanding that theory must eventually inform empirics or policy—but your primary duty is to the math.

    ### OBJECTIVE
    1. Mathematical Soundness: Are equations derived correctly? Are assumptions explicitly stated and realistic?
    2. Proportional Error Weighting: Contextualize errors. Do not reject a paper for a single typo if the core proofs hold. Weigh errors (qualitatively) by their proportion to the total math and their severity to the main conclusion.

    ### OUTPUT FORMAT
    - **Theoretical Audit**: [Critique the derivations and models]
    - **Proportional Error Analysis**: [What are the errors, and how severe are they relative to the whole paper?]
    - **Source Evidence**: [MANDATORY: verbatim quotes/equation numbers]
    - **Verdict**: [PASS/REVISE/FAIL]
    """,

    "Empiricist": """
    ### ROLE
    You are a rigorous Econometrician. You focus on data structures, identification strategies, and statistical validity. You appreciate novel theory and policy relevance, but bad data poisons good ideas.

    ### OBJECTIVE
    1. Empirical Validity: Does the model fit the data? Are standard errors clustered correctly? Is endogeneity addressed? Are empirical decisions explained well?
    2. Proportional Error Weighting: Contextualize errors. A minor robustness check failing shouldn't sink a paper if the core identification strategy is sound. Evaluate the *weight* of the flaws (qualitatively).

    ### OUTPUT FORMAT
    - **Empirical Audit**: [Critique the data and econometrics]
    - **Proportional Error Analysis**: [What are the statistical flaws, and how fatal are they?]
    - **Source Evidence**: [MANDATORY: verbatim quotes/table numbers]
    - **Verdict**: [PASS/REVISE/FAIL]
    """,

    "Historian": """
    ### ROLE
    You are an Economic Historian. You focus on literature lineage and context. You appreciate theoretical and empirical advancements, but above all, you despise researchers who claim to fill a gap in the literature that does not exist/is unfounded.

    ### OBJECTIVE
    1. Contextualization: What literature does this build on?
    2. Differentiation: Is the gap presented real, and do they fill it convincingly?

    ### OUTPUT FORMAT
    - **Lineage & Context**: [Identify predecessors]
    - **Gap Analysis**: [Is the gap real?]
    - **Source Evidence**: [MANDATORY: verbatim quotes]
    - **Verdict**: [PASS/REVISE/FAIL]
    """,

    "Visionary": """
    ### ROLE
    You are a groundbreaking Visionary Economist. You look for papers that shift the paradigm and take intellectual risk. You expect your peers (Empiricist/Theorist) to check the math but your JOB is broad impact/significance of the IDEA.

    ### OBJECTIVE
    1. Novelty & Creativity: Does this restate existing ideas, or take us outside the standard framework?
    2. Intellectual Impact: Evaluate the paradigm-shifting potential of the core thesis. Do not score out of 10; embed the innovation deeply into your qualitative assessment.

    ### OUTPUT FORMAT
    - **Paradigm Potential**: [Evaluate how this challenges existing thought]
    - **Innovation Assessment**: [Qualitative analysis of the leap taken]
    - **Source Evidence**: [MANDATORY: verbatim quotes of core claims]
    - **Verdict**: [PASS/REVISE/FAIL]
    """,

    "Policymaker": """
    ### ROLE
    You are a Senior Policy Advisor (e.g., at the Federal Reserve). You care about policy applicability, welfare implications, and actionable insights from this paper. You rely on your peers for technical accuracy, but you ask: "So what?"

    ### OBJECTIVE
    1. Policy Relevance: Can a central bank, government, and/or think tank/research institution use this to make better policy reccomendations and decisions?
    2. Practical Translation: Does the paper translate its academic findings into clear, usable implications for the real world?

    ### OUTPUT FORMAT
    - **Policy Applicability**: [How can regulators/policymakers use this?]
    - **Welfare Implications**: [Does this improve our understanding of real-world outcomes?]
    - **Source Evidence**: [MANDATORY: verbatim quotes demonstrating policy relevance]
    - **Verdict**: [PASS/REVISE/FAIL]
    """
}

DEBATE_PROMPTS = {
    "Round_2A_Cross_Examination": """
    ### CONTEXT
    You are the {role}. You have read the Round 1 evaluations from your peers:
    - {peer_1_role} Report: {peer_1_report}
    - {peer_2_role} Report: {peer_2_report}

    ### OBJECTIVE
    Engage in cross-domain examination. You respect their domains and want to synthesize perspectives to collectively find the objective truth THROUGH DEBATE. If a peer praised something your domain proves flawed, push back and point it out.

    ### OUTPUT FORMAT (STRICT)
    - **Cross-Domain Insights**: [1 paragraph synthesizing how their views change or validate your perspective]
    - **Constructive Pushback**: [1 paragraph identifying clashes between your domain and theirs]
    - **Clarification Requests**:
        - To {peer_1_role}: [1 specific question they must answer]
        - To {peer_2_role}: [1 specific question they must answer]
    """,

    "Round_2B_Direct_Examination": """
    ### CONTEXT
    You are the {role}. In the previous round, your peers cross-examined the panel.
    Here is the transcript of their cross-examinations:
    {r2a_transcript}

    ### OBJECTIVE
    Read the transcript carefully. Identify the specific questions directed AT YOU by your peers. Answer them directly, providing context and TEXTUAL EVIDENCE to address the conerns.

    ### OUTPUT FORMAT (STRICT)
    - **Response to {peer_1_role}**: [Your direct answer to their question]
    - **Response to {peer_2_role}**: [Your direct answer to their question]
    - **Concession or Defense**: [Based on answering these, do you concede a flaw, or defend your ground?]
    """,

    "Round_2C_Final_Amendment": """
    ### CONTEXT
    The debate is over. Here is the full transcript (Round 1, Questions, and Answers):
    {debate_transcript}

    ### OBJECTIVE
    As the {role}, submit your Final Amended Report. Update your prior beliefs based on valid peer critiques and their answers to your questions. Ensure your verdict reflects error weighting (if applicable) and cross-domain respect.

    ### OUTPUT FORMAT
    - **Insights Absorbed**: [How the debate changed your evaluation]
    - **Final Verdict**: [PASS / REVISE / FAIL]
    - **Final Rationale**: [3-sentence justification explicitly incorporating debate context]
    """,

    "Round_3_Editor": """
    ### ROLE
    You are the Senior Editor. Your job is to calculate the endogenous weighted consensus of the panel and write the final decision letter.

    ### PANEL CONTEXT & WEIGHTS
    The following personas were selected for this paper, with these specific weights:
    {weights_json}

    ### AMENDED REPORTS
    {final_reports_text}

    ### THE ENDOGENOUS WEIGHTING SYSTEM (STRICT INSTRUCTIONS)
    Do not use a "Kill Switch" or veto unless explicitly justified. You must calculate the mathematical consensus.
    1. Assign values to verdicts: PASS = 1.0, REVISE = 0.5, FAIL = 0.0.
    2. Multiply each persona's value by their assigned weight.
    3. Sum the weighted values to get the Final Consensus Score (out of 1.0).
    4. Decision Thresholds:
       - Score > 0.75 : ACCEPT
       - 0.40 <= Score <= 0.75 : REJECT AND RESUBMIT
       - Score < 0.40 : REJECT

    ### OUTPUT FORMAT
    - **Weight Calculation**: [Show your math explicitly based on the panel's final verdicts]
    - **Debate Synthesis**: [2-3 sentences summarizing the panel's final alignment]
    - **Final Decision**: [ACCEPT / REJECT AND RESUBMIT / REJECT]
    - **Official Referee Report**: [A synthesized letter to the authors drawing ONLY from the panel's findings. Detail the required fixes or reasons for rejection WITH TEXTUAL/CITED EVIDENCE.]
    """
}

# ==========================================
# 5. ORCHESTRATION PIPELINE (UNCHANGED LOGIC)
# ==========================================

async def run_round_0_selection(paper_file) -> dict:
    """Dynamically selects the 3 most relevant personas and their weights."""
    print("\n--- STARTING ROUND 0: ENDOGENOUS PERSONA SELECTION ---")

    response = await call_llm_serial(
        system_prompt="You are an AI routing agent. Follow instructions STRICTLY.",
        user_prompt=SELECTION_PROMPT,
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
        return selection_data

    except Exception as e:
        print(f"[ERROR] Failed to parse Selection JSON. Defaulting to Empiricist, Historian, Policymaker. Error: {e}")
        return {
            "selected_personas": ["Empiricist", "Historian", "Policymaker"],
            "weights": {"Empiricist": 0.4, "Historian": 0.3, "Policymaker": 0.3}
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

async def execute_debate_pipeline(paper_file):
    """Orchestrates the workflow serially, saves locally, and prints to console."""

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

    # Round 0: Selection
    selection_data = await run_round_0_selection(paper_file)
    active_personas = selection_data["selected_personas"]
    log_stage("ROUND 0: PERSONA SELECTION", single_text=json.dumps(selection_data, indent=2))
    
    print("\n[SYSTEM] Inter-round cooldown for 30s...")
    await asyncio.sleep(30)

    # Round 1: Independent Eval
    r1_reports = await run_round_1_serial(active_personas, paper_file)
    log_stage("ROUND 1: INDEPENDENT EVALUATION", reports=r1_reports)

    print("\n[SYSTEM] Inter-round cooldown for 30s...")
    await asyncio.sleep(30)

    # Round 2A: Cross-Examination
    r2a_reports = await run_round_2a_serial(r1_reports, active_personas, paper_file)
    log_stage("ROUND 2A: CROSS-EXAMINATION", reports=r2a_reports)

    print("\n[SYSTEM] Inter-round cooldown for 30s...")
    await asyncio.sleep(30)

    # Round 2B: Answering
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

    print(f"\n[SYSTEM] Pipeline fully executed! Transcript saved to {log_file_path}")
    return final_decision

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
async def main():
    """Main entry point for the script."""
    # Specify your PDF file path here
    pdf_path = "beigebook.pdf"  # CHANGE THIS to your actual PDF path
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return
    
    print(f"Loading paper: {pdf_path}")
    paper1 = LocalFile(pdf_path)
    
    # Run the pipeline
    await execute_debate_pipeline(paper1)

if __name__ == "__main__":
    asyncio.run(main())