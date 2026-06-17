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

OUTPUT_DIR = #insert file path here
os.makedirs(OUTPUT_DIR, exist_ok=True)
ACTIVE_MODEL = "gemini-3.5-flash"

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

        # Extract only strict debate tags to compress debate logs and avoid 
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
def calculate_final_score(final_reports, weights_dict, mode="weighted"):
    """
    Python handles the math so the LLM doesn't hallucinate.
    """
    scores = {"PASS": 1.0, "REVISE": 0.5, "FAIL": 0.0}

    if mode == "weighted":
        final_score = 0.0
        for role, text in final_reports.items():
            verdict = extract_verdict(text) # e.g., gets "PASS"
            final_score += scores[verdict] * weights_dict[role]
        return final_score

    # (Other modes like 'voting' or 'logical_graph' go here)
