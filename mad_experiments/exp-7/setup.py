# SETUP
import time
import os
import asyncio
import datetime
import json
import re
from functools import partial
from google import genai
from google.genai import errors, types
from google.colab import drive, userdata
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

drive.mount('/content/drive')
API_KEY = userdata.get('GEMINI_API_KEY')
client = genai.Client(api_key=API_KEY)

OUTPUT_DIR = #insert directory where you want outputs here 
os.makedirs(OUTPUT_DIR, exist_ok=True)
ACTIVE_MODEL = "gemini-3.5-flash" #replace with model of choice

FALLBACK_MODELS = [
    "gemini-3.5-flash-preview",
    "gemini-3.1-flash-lite"
]

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((errors.APIError, Exception))
)
def generate_safe_content(system_prompt: str, user_prompt: str, role: str, files=None, temperature: float = 0.0):
    global ACTIVE_MODEL, FALLBACK_MODELS
    print(f"[{role}] Generating response using {ACTIVE_MODEL} (Temp: {temperature})...")

    content_list = [user_prompt]
    if files:
        content_list.extend(files)

    # Global safety configuration to minimize false-positive blocks on long text
    lenient_safety = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        )
    ]

    try:
        response = client.models.generate_content(
            model=ACTIVE_MODEL,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                safety_settings=lenient_safety
            )
        )

        # Guard against API 200 Successes that return empty payloads due to internal structural blocks
        if not response or not response.text:
            # Look inside candidates metadata to extract block warnings
            finish_reason = "UNKNOWN"
            if response and response.candidates:
                finish_reason = getattr(response.candidates[0], "finish_reason", "UNKNOWN")

            print(f"  [⚠️ WARNING] Empty text returned from API. Finish Reason: {finish_reason}")

            # If it's a structural safety block, fallback to a string format so the downstream parser doesn't crash
            if finish_reason == "SAFETY":
                return '{"selected_personas": [], "weights": {}, "justification": "Block: Document structure flagged by built-in endpoint filters."}'
            raise ValueError("Empty response payload received from endpoint.")

        return response.text

    except Exception as e:
        msg = str(e).upper()

        if "429" in msg or "UNAVAILABLE" in msg:
            print(f"  [API Traffic] Delaying request on {ACTIVE_MODEL}...")
            raise e

        elif "500" in msg or "503" in msg or "QUOTA" in msg:
            if FALLBACK_MODELS:
                print(f"\n[🚨 ALERT] {ACTIVE_MODEL} failed. Rerouting to {FALLBACK_MODELS[0]}...")
                ACTIVE_MODEL = FALLBACK_MODELS.pop(0)
                raise e
            else:
                print("  [Fatal Error] All fallback models exhausted.")
                raise e
        else:
            print(f"  [Error] {msg}")
            raise e

async def call_llm_serial(system_prompt: str, user_prompt: str, role: str, files=None, temperature: float = 0.0) -> str:
    func = partial(generate_safe_content, system_prompt, user_prompt, role, files, temperature=temperature)
    return await asyncio.to_thread(func)

# Helper functions for debate
# Summarization between rounds
def compress_transcript(reports_dict):
    """
    Strips conversational filler and extracts only the hard logic tags
    to feed into the next round.
    """
    compressed_text = ""
    for role, text in reports_dict.items():
        compressed_text += f"\n--- {role.upper()} LOGIC STATE ---\n"

        # Extract only the strict formatting tags we defined
        tags = re.findall(r"(\[(?:ATTACK|DEFEND|CONCEDE|QUESTION).*?\](?:.*?)(?=\n\[|$))", text, re.DOTALL)

        if tags:
            compressed_text += "\n".join(tags)
        else:
            compressed_text += "[No formal logic tags issued in this round.]"

    return compressed_text

def extract_verdict(text):
    match = re.search(r"(?i)\b(PASS|REVISE|FAIL)\b", text)
    if match:
        return match.group(1).upper()
    # If no verdict found, assume REVISE (the most conservative path)
    return "REVISE"

# Calculation tool for editor
import re

def calculate_final_score(final_reports, weights_dict, debate_history=None, mode="probabilistic_dung"):
    """
    Calculates the final outcome based on three distinct mechanical architectures.
    Ensures a valid float is always returned.
    """
    # Normalize mode to lowercase to avoid string match issues
    mode = str(mode).strip().lower()

    # Define a generic score mapping for classic modes
    scores = {"PASS": 1.0, "REVISE": 0.5, "FAIL": 0.0}

    if mode == "voting":
        total = sum(scores.get(extract_verdict(text), 0.0) for text in final_reports.values())
        return float(total / len(final_reports))

    elif mode == "probabilistic_dung":
        # Start with a baseline score (perfect paper score)
        paper_score = 1.0
        full_final_text = "\n".join(final_reports.values())

        # Flexibly match attack tags with or without brackets, capture severity and confidence
        # Matches formats like: [ATTACK: Econometrician] ... Δ-High ... Confidence 8
        attack_pattern = r"(?:\[?ATTACK\s*:\s*[^\]\n]+\]?)(.*?)(?=Delta|Δ|Confidence|\[|\]|$)"

        # Let's clean and find active markers across the aggregated final text
        attacks = re.finditer(r"\[ATTACK:[^\]]+\]", full_final_text, re.IGNORECASE)
        has_attacks = False

        # Parse text line by line or block by block for structural metrics
        # If your models change text structure, this fallback protects the loop
        lines = full_final_text.split("\n")
        for line in lines:
            if "[ATTACK" in line.upper():
                has_attacks = True
                # Extract numerical confidence (defaults to 5 if model omitted it)
                conf_match = re.search(r"(?i)Confidence\s*[:\s]*(\d+(?:\.\d+)?)", line)
                confidence = float(conf_match.group(1)) if conf_match else 5.0

                # Extract severity delta
                severity = "LOW"
                if "HIGH" in line.upper() or "Δ-HIGH" in line.upper():
                    severity = "HIGH"
                elif "MEDIUM" in line.upper() or "Δ-MEDIUM" in line.upper():
                    severity = "MEDIUM"

                P_attack = confidence / 10.0

                # Attenuate if explicitly defended in proximity
                if "DEFEND" in line.upper() or "DEFEND" in full_final_text:
                    P_attack *= 0.5
                if "CONCEDE" in line.upper():
                    P_attack = 1.0

                # Scale penalty based on validated severity
                if severity == "HIGH":
                    penalty = P_attack * 0.40
                elif severity == "MEDIUM":
                    penalty = P_attack * 0.20
                else:
                    penalty = P_attack * 0.05

                paper_score -= penalty

        # If no explicit attacks were successfully parsed but text existed,
        # evaluate the fallback to the baseline weighted average score
        if not has_attacks:
            total_weighted = 0.0
            for role, text in final_reports.items():
                total_weighted += scores.get(extract_verdict(text), 0.5) * weights_dict.get(role, 1.0/len(final_reports))
            return float(total_weighted)

        return float(max(0.0, min(1.0, paper_score)))

    else:
        # ORIGINAL / DEFAULT: Weighted Verdict
        final_score = 0.0
        for role, text in final_reports.items():
            verdict = extract_verdict(text)
            final_score += scores.get(verdict, 0.5) * weights_dict.get(role, 1.0 / len(final_reports))
        return float(final_score)


def extract_round_1_essentials(text):
    """Pulls only the Domain Audit and Severity Delta from Round 1."""
    audit_match = re.search(r"(?i)\*\*Domain Audit\*\*:\s*(.*?)(?=\n\*\*|$)", text, re.DOTALL)
    severity_match = re.search(r"(?i)\*\*Severity Delta.*?\*\*:\s*(.*?)(?=\n\*\*|$)", text, re.DOTALL)

    audit = audit_match.group(1).strip() if audit_match else "[Audit not found]"
    severity = severity_match.group(1).strip() if severity_match else "[Severity not found]"

    return f"- Domain Audit: {audit}\n- Severity: {severity}"

def extract_debate_essentials(text):
    """Pulls logic tags and Final Argument State for Loop 2+."""
    tags = re.findall(r"(\[(?:ATTACK|DEFEND|CONCEDE|QUESTION).*?\](?:.*?)(?=\n\[|\n\*\*|$))", text, re.DOTALL)
    final_state_match = re.search(r"(?i)\*\*Final Argument State\*\*:\s*(.*?)(?=\n\*\*|$)", text, re.DOTALL)

    output = "\n".join(tags).strip() if tags else "[No active attacks or concessions.]"
    if final_state_match:
        output += f"\n\n- Final Argument State: {final_state_match.group(1).strip()}"

    return output
