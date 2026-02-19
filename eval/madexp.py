import time
import os
import asyncio
import datetime
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
OUTPUT_DIR = "/content/drive/MyDrive/Senior Year/Fed Stuff/LLM_Peer_Review_Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. ROBUST LLM CALLS (SYNC & ASYNC)
# ==========================================
@retry(
    stop=stop_after_attempt(7), 
    wait=wait_exponential(multiplier=2, min=10, max=120), 
    retry=retry_if_exception_type(errors.ClientError)
)
def generate_safe_content(system_prompt: str, user_prompt: str, files=None):
    """Synchronous Gemini call with system instructions and files."""
    try:
        content_list = [user_prompt]
        if files:
            content_list.extend(files)
            
        response = client.models.generate_content(
            model="gemini-2.5-flash", #update based on your token limits and model novelty
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.5 
            )
        )
        return response.text
    except errors.ClientError as e:
        if "429" in str(e): 
            print("  [API] Rate limit (429) reached. Tenacity is backing off and retrying...")
            raise e 
        elif "503" in str(e):
            print("  [API] Server overloaded (503). Retrying...")
            raise e
        else:
            raise e

async def call_llm_async(system_prompt: str, user_prompt: str, role: str, files=None) -> str:
    """Async wrapper with STAGGERING to avoid hitting Free Tier burst limits.""" #can remove if on higher paid tier 
    stagger_map = {"Mathematician": 0, "Historian": 15, "Visionary": 30, "Editor": 0}
    delay = stagger_map.get(role, 0)
    
    if delay > 0:
        print(f"[{role}] Queued. Waiting {delay}s to avoid rate limit...")
        await asyncio.sleep(delay)
        
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
# 3. FILE UPLOADS
# ==========================================
# UNCOMMENT THESE WHEN READY TO RUN
paper1 = client.files.upload(file='/content/drive/MyDrive/Senior Year /Fed Stuff/MadTest1.pdf') 
wait_for_files_active([paper1])
print(f"Uploaded {paper1.display_name} as {paper1.name}")

# ==========================================
# 4. PROMPTS (Kept exactly as you defined them)
# ==========================================
SYSTEM_PROMPTS = {
    "Mathematician": """
    ### ROLE
    You are a harsh technical specialist that focuses on methodological, mathematical, and econometric validity. You ensure derivations are accurate, empirical estimates are valid, and details are flawless. You do not care about the "story" or societal impact.

    ### OBJECTIVE
    STEP 1: CLASSIFY THE PAPER.
    Determine if the paper is [THEORETICAL] (focuses on proofs/models without data), [EMPIRICAL] (focuses on data analysis/econometrics), or [MIXED]. Do not penalize a theoretical paper for lacking empirical data, and do not penalize an applied empirical paper for lacking novel mathematical proofs.

    STEP 2: EVALUATE BASED ON CLASSIFICATION.
    1. Methodological soundness: Is the model laid out appropriate to the research question? 
    2. Empirical validity (If Empirical/Mixed): Does the model fit the data structure? Is it well specified? Are coefficients reported and interpreted correctly?
    3. Mathematical accuracy (If Theoretical/Mixed): Are equations presented and solved correctly? Are assumptions explicit and realistic?

    ### OUTPUT FORMAT
    - **Paper Archetype**: [Theoretical / Empirical / Mixed]
    - **Methodology & Math Audit**: [Critique the model choice, derivations, or statistics based on the archetype]
    - **Source Evidence**: [MANDATORY: Provide verbatim quotes, table references, or exact equation numbers to justify *every* flaw identified. If you cannot quote it, do not claim it.]
    - **Verdict**: [PASS/FAIL] 
        - PASS Criteria: The math/derivations are sound, assumptions are explicit, and/or the empirical model matches the data structure.
        - FAIL Criteria: Auto-fail for fundamental mathematical inaccuracy, ignored endogeneity, or omitted variable bias.
    """,

    "Historian": """
    ### ROLE
    You are the most knowledgeable economic historian in the galaxy. You have read every paper in this field from the last 100 years. You despise contextual inaccuracy and researchers who claim they are the first to discover something that was solved decades ago.

    ### OBJECTIVE
    1. Contextualization: What literature does this paper build off of? Does it come out of left field, or is it firmly grounded in the field's lineage?
    2. Recency vs. Endurance: How up-to-date are the citations relative to the publication date? Does this paper have enduring relevance?
    3. Differentiation: How well does the paper identify a true gap in the literature, and does it fill it?

    ### OUTPUT FORMAT
    - **Lineage & Context**: [Identify the predecessors of this work]
    - **Gap Analysis**: [Is the gap presented real and do the authors actually fill that gap?]
    - **Source Evidence**: [MANDATORY: Provide verbatim quotes from the paper's literature review or introduction that demonstrate false novelty, missing context, or strong contextualization.]
    - **Verdict**: [PASS/FAIL]
        - PASS Criteria: Accurately cites foundational texts, addresses recent advancements, clearly distinguishes its contribution.
        - FAIL Criteria: Auto-fail if it claims false novelty (ignoring older papers), suffers from extreme recency bias, or fails to anchor assumptions.
    """,

    "Visionary": """
    ### ROLE
    You are a groundbreaking, Nobel-Prize winning economist who values novelty above all else. You only care about papers that have the potential to change the field. Assume the math is correct (the Mathematician will check that). You care purely about the idea.

    ### OBJECTIVE
    1. Novelty of contribution: How new is this contribution? Does it merely restate existing ideas, or does it make a novel argument? 
    2. Creativity: Does the main thesis follow a predictable line of reasoning, or does it approach the problem from a contextually interesting angle?
    3. Scope: What is the potential impact of this paper on the field, related disciplines, and broader society/culture?

    ### OUTPUT FORMAT
    - **Innovation Score**: [1-10 scale assessing pure novelty]
    - **Paradigm Potential**: [Evaluate how this challenges or changes existing thought]
    - **Source Evidence**: [MANDATORY: Provide verbatim quote(s) of the paper's core thesis or main claim that you are evaluating for novelty.]
    - **Verdict**: [PASS/FAIL]
        - PASS Criteria: The core thesis is a genuine leap forward. It takes intellectual risks or connects disparate fields.
        - FAIL Criteria: Auto-fail for incrementalism.
    """
}

DEBATE_PROMPTS = {
    "Round_2A_Cross_Examination": """
    ### CONTEXT
    You are the {role}. You have just read the Round 1 evaluations from your peers:
    - {peer_1_role} Report: {peer_1_report}
    - {peer_2_role} Report: {peer_2_report}

    ### OBJECTIVE
    Engage in a constructive cross-domain examination. You are not here to blindly defend your turf; you are here to synthesize their insights with your domain expertise to find the objective truth about this paper.
    1. Identify Cross-Domain Impacts: Does the {peer_1_role}'s critique fundamentally change how you view the paper? (e.g., If the data is flawed, does the "novel" idea still matter?)
    2. Constructive Friction: Where do your domains clash? If a peer praised something you found lacking, gently and rigorously point out the discrepancy using exact quotes from the text with the goal of resolution.
    3. Information Gaps: What specific evidence do you need from the other reviewers to finalize your assessment? Ask them directly.

    ### OUTPUT FORMAT
    - **Synthesis with {peer_1_role}**: [1 paragraph: What valid points do you accept from their report, and how does it nuance your view?]
    - **Synthesis with {peer_2_role}**: [1 paragraph: What valid points do you accept from their report, and how does it nuance your view?]
    - **Constructive Pushback**: [1 paragraph: Where do you disagree with their assessments based on your domain rules?]
    - **Clarification Requests**: [1-2 specific questions or text-referencing demands your peers must address in their final update]
    """,

    "Round_2B_Final_Amendment": """
    ### CONTEXT
    The cross-examination phase is over. Here is the debate transcript:
    {debate_transcript}

    ### OBJECTIVE
    As the {role}, you must now submit your Final Amended Report to the Senior Editor. Reflect sincerely on the debate. You are highly encouraged to update your prior beliefs based on valid peer critiques—this is a sign of intellect and rigorous peer editing, not weakness. Integrate their insights, but ensure your core domain requirements are met.

    ### OUTPUT FORMAT
    - **Insights Absorbed**: [Explicitly state how the other reviewers changed or improved your initial evaluation]
    - **Remaining Domain Friction**: [What fundamental issues from your domain remain unresolved or unaddressed?]
    - **Final Verdict**: [PASS / REVISE / FAIL]
        - *PASS*: Meets all domain standards.
        - *REVISE*: Core idea/math/context is solid, but requires specific, fixable adjustments.
        - *FAIL*: Fundamentally broken in your domain; cannot be salvaged by a rewrite.
    - **Final Rationale**: [A definitive 3-sentence justification for your verdict, explicitly incorporating how the debate influenced this final stance]
    """,
    
    "Round_3_Editor": """
    ### ROLE
    You are the Senior Editor of a top-tier economics journal. You are decisive, fair, and ruthless. You do not invent new critiques; your sole job is to synthesize the final reports from your review panel and enforce the journal's standards.

    ### CONTEXT
    Here are the Final Amended Reports from your panel:
    - Mathematician: {math_final_report}
    - Historian: {historian_final_report}
    - Visionary: {visionary_final_report}

    ### HIERARCHY OF TRUTH (STRICT RULES)
    1. **The Technical Kill Switch (FAIL)**: If the Mathematician marks [FAIL] for fatal empirical/math errors, or the Historian marks [FAIL] for completely derivative work, the paper is REJECTED. Bad data or repetitive ideas should not be salvaged.
    2. **The Incremental Trap (FAIL)**: If both Math and History are fine, but the Visionary marks [FAIL] (too incremental), the paper is REJECTED.
    3. **The Salvage Pathway (REJECT AND RESUBMIT)**: If the Visionary marks [PASS] (highly novel idea) or the Historian marks [PASS] (highly relevant), but ANY reviewer marks [REVISE] for fixable errors (e.g., missing citations, fixable omitted variable bias, formatting), the decision is REJECT AND RESUBMIT. 
    4. **The Clear (ACCEPT)**: If all three mark [PASS] the decision is ACCEPT.

    ### OUTPUT FORMAT
    - **Debate Synthesis**: [2-3 sentences summarizing the panel's final alignment, explicitly noting what priors were updated during the debate]
    - **Hierarchy Application**: [Explicitly state how the Hierarchy of Truth was applied to the [PASS/REVISE/FAIL] verdicts to reach the decision]
    - **Final Decision**: [ACCEPT / REJECT AND RESUBMIT / REJECT]
    - **Official Referee Report**: [A professional, synthesized letter to the authors outlining the primary reasons for the decision. If REJECT AND RESUBMIT, provide a bulleted list of mandatory fixes. Draw ONLY from the panel's explicit findings and quotes.]
    """
}

# ==========================================
# 5. ORCHESTRATION PIPELINE
# ==========================================
async def run_round_1(paper_file) -> dict:
    user_prompt = "Please read the attached paper and provide your Round 1 evaluation based on your role."
    tasks = {
        "Mathematician": call_llm_async(SYSTEM_PROMPTS["Mathematician"], user_prompt, "Mathematician", files=[paper_file]),
        "Historian": call_llm_async(SYSTEM_PROMPTS["Historian"], user_prompt, "Historian", files=[paper_file]),
        "Visionary": call_llm_async(SYSTEM_PROMPTS["Visionary"], user_prompt, "Visionary", files=[paper_file])
    }
    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))

async def run_round_2a(r1_reports: dict, paper_file) -> dict:
    tasks = {}
    for role in ["Mathematician", "Historian", "Visionary"]:
        peers = [p for p in r1_reports.keys() if p != role]
        prompt_2a = DEBATE_PROMPTS["Round_2A_Cross_Examination"].format(
            role=role, peer_1_role=peers[0], peer_1_report=r1_reports[peers[0]],
            peer_2_role=peers[1], peer_2_report=r1_reports[peers[1]]
        )
        tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], prompt_2a, role, files=[paper_file])
    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))

async def run_round_2b(r1_reports: dict, r2a_reports: dict, paper_file) -> dict:
    transcript = "ROUND 1 REPORTS:\n"
    for role, report in r1_reports.items():
        transcript += f"\n[{role}]:\n{report}\n"
    transcript += "\nCROSS-EXAMINATION:\n"
    for role, report in r2a_reports.items():
        transcript += f"\n[{role}]:\n{report}\n"

    tasks = {}
    for role in ["Mathematician", "Historian", "Visionary"]:
        prompt_2b = DEBATE_PROMPTS["Round_2B_Final_Amendment"].format(
            role=role, debate_transcript=transcript
        )
        tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], prompt_2b, role, files=[paper_file])
    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))

async def run_round_3(r2b_reports: dict) -> str:
    prompt_3 = DEBATE_PROMPTS["Round_3_Editor"].format(
        math_final_report=r2b_reports["Mathematician"],
        historian_final_report=r2b_reports["Historian"],
        visionary_final_report=r2b_reports["Visionary"]
    )
    return await call_llm_async(
        system_prompt="You are the Senior Editor. Follow the instructions provided.", 
        user_prompt=prompt_3, role="Editor"
    )

async def execute_debate_pipeline(paper_file):
    """Orchestrates the workflow, saves to Drive, and prints to console."""
    
    # 1. Setup the File Logger
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = paper_file.display_name.replace('.pdf', '') if paper_file.display_name else "Unknown_Paper"
    log_file_path = os.path.join(OUTPUT_DIR, f"Review_{safe_name}_{timestamp}.txt")
    
    print(f"\n[SYSTEM] Saving live outputs to: {log_file_path}\n")

    def log_stage(stage_title, reports=None, single_text=None):
        """Helper function to print to the screen AND write to the text file."""
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

    # Step 1: Independent Eval
    print("\n--- STARTING ROUND 1: INDEPENDENT EVALUATION ---")
    r1_reports = await run_round_1(paper_file)
    log_stage("ROUND 1: INDEPENDENT EVALUATION", reports=r1_reports)
    
    print("\n[SYSTEM] Round 1 Complete. Cooling down for 45s...")
    await asyncio.sleep(45) 
    
    # Step 2A: Debate Part A
    print("\n--- STARTING ROUND 2A: CROSS-EXAMINATION ---")
    r2a_reports = await run_round_2a(r1_reports, paper_file)
    log_stage("ROUND 2A: CROSS-EXAMINATION", reports=r2a_reports)
    
    print("\n[SYSTEM] Round 2A Complete. Cooling down for 45s...")
    await asyncio.sleep(45)
    
    # Step 2B: Debate Part B
    print("\n--- STARTING ROUND 2B: FINAL AMENDMENTS ---")
    r2b_reports = await run_round_2b(r1_reports, r2a_reports, paper_file)
    log_stage("ROUND 2B: FINAL AMENDMENTS", reports=r2b_reports)
    
    print("\n[SYSTEM] Round 2B Complete. Cooling down for 15s...")
    await asyncio.sleep(15)
    
    # Step 3: Final Synthesis
    print("\n--- STARTING ROUND 3: EDITOR DECISION ---")
    final_decision = await run_round_3(r2b_reports)
    log_stage("FINAL REFEREE REPORT (EDITOR)", single_text=final_decision)
    
    print(f"\n[SYSTEM] Pipeline fully executed! Transcript safely saved to your Google Drive at {OUTPUT_DIR}")
    
    return final_decision

# ==========================================
# 6. EXECUTION 
# ==========================================

await execute_debate_pipeline(paper1) #replace with desired paper
