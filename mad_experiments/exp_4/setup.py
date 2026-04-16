#Setup
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

#Change this for claude 
drive.mount('/content/drive')
API_KEY = userdata.get('GEMINI_API_KEY')
client = genai.Client(api_key=API_KEY)

#Change this for linux 
OUTPUT_DIR = "/content/drive/MyDrive/SENIOR_YEAR/FED"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ACTIVE_MODEL = "gemini-2.5-flash"

#Only uncomment if necessary to use other models (if you face RPM/RPD limits)
#FALLBACK_MODELS = []

#Only uncomment if you face RPM limits and need API request cooldowns 
#@retry(
#    stop=stop_after_attempt(12), 
#    wait=wait_exponential(multiplier=10, min=30, max=300), 
#    retry=retry_if_exception_type((errors.ClientError, Exception))
#)

#LLM querying function 
def generate_safe_content(system_prompt: str, user_prompt: str, role: str, files=None):
    global ACTIVE_MODEL, FALLBACK_MODELS
    # MOVE THE PRINT STATEMENT HERE so it fires on every retry!
    print(f"[{role}] Generating response using {ACTIVE_MODEL}...")
    
    try:
        content_list = [user_prompt]
        if files: content_list.extend(files)

        response = client.models.generate_content(
            model=ACTIVE_MODEL,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt, temperature=0.4
            )
        )
        return response.text
    except Exception as e:
        msg = str(e).upper()
        if "QUOTA" in msg and ("DAY" in msg or "EXCEEDED" in msg):
            if FALLBACK_MODELS:
                print(f"\n[🚨 QUOTA] {ACTIVE_MODEL} exhausted. Rerouting to {FALLBACK_MODELS[0]}...")
                ACTIVE_MODEL = FALLBACK_MODELS.pop(0)
                raise e 
            else:
                raise Exception("All fallback models exhausted for the day.")
        elif "429" in msg or "503" in msg or "UNAVAILABLE" in msg:
            print(f"  [API Traffic] Delaying request on {ACTIVE_MODEL}...")
            raise e
        else:
            raise e

async def call_llm_serial(system_prompt: str, user_prompt: str, role: str, files=None) -> str:
    # We remove the print statement from here, but pass the role down
    return await asyncio.to_thread(generate_safe_content, system_prompt, user_prompt, role, files)

#Upload paper to generate referee report for here
paper1 = client.files.upload(file='')

def wait_for_files_active(files):
    print("Verifying file status...")
    for f in files:
        file_status = client.files.get(name=f.name)
        while file_status.state.name == "PROCESSING":
            time.sleep(2)
            file_status = client.files.get(name=f.name)
        if file_status.state.name == "FAILED": raise Exception("File failed to process.")
    print(f"  ✓ {files[0].display_name} is ACTIVE.")

wait_for_files_active([paper1])

#Run phase 1 if you want to ensure that your paper has been correctly uploaded to the terminal 
# --- PHASE 1: SANITY CHECK ---
#print("\n--- RUNNING PDF EXTRACTION SANITY CHECK ---")
#sanity_prompt = """
#Please extract and return ONLY the following three items from the document:
#1. The EXACT first 2 sentences of the Abstract.
#2. The exact title/caption of Table 1 (or Figure 1 if no tables exist).
#3. The exact final concluding sentence of the paper (before the references/#appendix).
#4. A summary of the paper's purpose, findings, and significance.
#"""
#sanity_result = generate_safe_content("You are a precise data extraction tool.#", sanity_prompt, [paper1])
#print(sanity_result)
#print("\n[VERIFY THIS OUTPUT AGAINST YOUR PDF BEFORE PROCEEDING]")

