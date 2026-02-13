#Copy and pasted from ideation agent instead of using architecture; have to change?
#This is where I want to create the overall infrastructure, and the app.py file will be for where I deploy and customize app stuff 
#First, I need to put the relevant stuff from architecture here, and I'll test out to see if the LLM call works on the streamlit (DONE)
#Then, I will put all the workflows in the app pyfile for now, maybe putting them in a different py file later and run and see how it works (have four workflows for now and can update later)

import requests
import json
import os
import sys
import importlib
import pandas as pd
import csv
import time 
import tiktoken 
import math 
from typing import List, Dict, Any 
from collections import deque 
from pathlib import Path
import logging
import pdfplumber

# API Configuration
API_KEY = "" # API key
API_BASE = ""  # API endpoint
url_chat_completions = f"{API_BASE}/chat/completions"
model_selection = ""        # fast + cheap
model_selection2 = ""            # strong general model
model_selection3 = ""           # stronger reasoning


def single_query(prompt: str, debug_flag=False, retries=3) -> str: 
    url = url_chat_completions
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_selection3,
        "messages": [{"role": "system", "content": """You are an advanced research assistant specializing in economics. Your expertise spans economics, computer science, data science, mathematics, statistics and all social sciences. You excel at synthesizing information across literature corpora, spotting gaps, trends, and insights, making inferences, and proposing creative solutions to research challenges. Your responses are informative, accessible, and tailored to the researcher's level of expertise.When an economist asks what you can do for them, clearly explain your four specialized workflows:
        1. Literature Relevance Analysis: Evaluate and funnel down papers based on relevance to a specific research focus. You'll flag which papers are most pertinent to their interests. (Note: requires input papers)
        2. Methodology-Driven Ideation: Generate novel research questions tailored to a specific methodological approach or empirical/theoretical framework the economist wants to employ.3. Policy Paper Inspiration: Derive research questions from Federal Reserve publications including FEDS notes, IFPD discussion papers, FOMC notes, and other Fed policy documents. (Note: requires input sources, whether policy exerpts or uploaded text)
        4. Cross-Paper Synthesis: Identify potential research directions by analyzing patterns, gaps, and emerging themes across multiple research papers, synthesizing insights to suggest promising new avenues of inquiry. (Note: requires input papers) After explaining these workflows, always ask: "Which of these approaches would be most helpful for your current research needs? Or would you like me to explain any of these workflows in more detail?" If you already have an idea of what route the economist wants you to pursue, confirm your intention and ask for necessary follow up information (e.g. files).Actively engage with the researcher, asking clarifying questions when needed to provide the most relevant and valuable assistance throughout the ideation process."""}, 
        {"role": "user", "content": prompt}], 
     #"thinking": {
     #   "type": "enabled",
     #   "budget_tokens": 2048   # allocate up to 2 048 tokens for internal reasoning
    #},
    "temperature": 0.5
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                if debug_flag:
                    print("Success!")

                output_text = response.json()
                return output_text["choices"][0]["message"]["content"]
            else:
                print(f"API Error {response.status_code}: {response.text}")
        except requests.RequestException as e:
            print(f"API request failed: {e}")

        if attempt < retries - 1:
            print("Retrying in 5 seconds...")
            time.sleep(5)

    return "API error"

try:
    ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        """
        Count tokens in text using tiktoken if available, otherwise use a simple approximation.
        Completely avoids the _ENC variable name issue.
        """
        if text is None:
            return 0
            
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Simple fallback: approximating tokens as 1.3 times the number of words
            return max(1, int(len(text.split()) * 1.3))
except Exception:
    def count_tokens(text: str) -> int:
    # if tiktoken isn't correctly imported, counts tokens by assigning 1.3 tokens per word + punctuation
        return max(1, int(len(text.split()) * 1.3))


class ConversationManager:
    def __init__(self, 
                 conversation_history: List[Dict[str, str]],
                 summarize_fn=None,
                 total_token_limit: int = 8000,
                 recent_token_budget: int = 1000, 
                 reply_tokens: int = 256, 
                 segment_size_msgs: int = 6):
        
        self.system_prompt = conversation_history[0]  # system message
        self.ch = conversation_history[1:]  # exclude system message from history
        self.summarize_fn = summarize_fn or self._fallback_summarize
        self.total_token_limit = total_token_limit
        self.recent_token_budget = recent_token_budget 
        self.reply_reserved_tokens = reply_tokens
        self.segment_size_msgs = segment_size_msgs

    def total_tokens(self) -> int:
        return sum(m.get("tokens", 0) for m in self.ch)

    def prune_if_needed(self):
        history_budget = self.total_token_limit - self.reply_reserved_tokens
        if self.total_tokens() <= history_budget:
            return

        raw_msgs = [m for m in self.ch if not m.get("is_summary", False)]
        kept_raw = []
        running = 0
        for m in reversed(raw_msgs):
            if running + m["tokens"] <= self.recent_token_budget:
                kept_raw.append(m)
                running += m["tokens"]
            else:
                break
        kept_raw_ids = set(id(m) for m in kept_raw)
        older_raw = [m for m in raw_msgs if id(m) not in kept_raw_ids]
        segments = [older_raw[i:i+self.segment_size_msgs] for i in range(0, len(older_raw), self.segment_size_msgs)]

        new_ch = []
        seg_msg_set = set()
        for seg in segments:
            for m in seg:
                seg_msg_set.add(id(m))
        seg_map = {}
        for seg in segments:
            if seg:
                seg_map[id(seg[0])] = seg

        skip_set = set()
        for m in self.ch:
            mid = id(m)
            if mid in skip_set:
                continue
            if mid in seg_map:
                seg = seg_map[mid]
                texts = [s["text"] for s in seg]
                summary_text = self.summarize_fn(texts)
                summary_msg = {
                    "role": "system",  # valid role!
                    "text": f"(Summary of previous context): {summary_text}",
                    "tokens": count_tokens(summary_text),
                    "ts": seg[-1]["ts"],
                    "is_summary": True
                }
                new_ch.append(summary_msg)
                for s in seg:
                    skip_set.add(id(s))
            else:
                new_ch.append(m)
        self.ch[:] = new_ch

    def conv_query(self, prompt: str,
                           debug_flag=False,
                           retries=3,
                           max_tokens: int = 500,
                           context_window: int = None) -> str: 
        url = url_chat_completions
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        self.append("user", prompt)
        self.prune_if_needed()

        data = {
            "model": model_selection,
            "messages": [self.system_prompt] + [
                {"role": m["role"], "content": m.get("text", m.get("content", ""))} for m in self.ch
            ],
            #"thinking": {
            #    "type": "enabled",
            #    "budget_tokens": 2048
            #},
            "temperature": 1
        }

        for attempt in range(retries):
            try:
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    if debug_flag:
                        print("Success!")
                    output_text = response.json()
                    assistant_response = output_text["choices"][0]["message"]["content"]
                    self.append("assistant", assistant_response)
                    return assistant_response
                else:
                    print(f"API Error {response.status_code}: {response.text}")
            except requests.RequestException as e:
                print(f"API request failed: {e}")
            if attempt < retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
        return "API error"

    def append(self, role: str, text: str):
        msg = {
            "role": role,
            "text": text,
            "tokens": count_tokens(text),
            "ts": time.time(),
            "is_summary": False
        }
        self.ch.append(msg)
        self.prune_if_needed()

    def clear_history(self):
        self.ch = []
        print("Done")
    
    def print_history(self):
        return list(self.ch)

    def set_system_prompt(self, prompt_text: str):
        self.system_prompt = {"role": "system", "content": prompt_text}

    def _fallback_summarize(self, texts: list) -> str:
        return "Summary not available (fallback function used)."


#Different system prompts: 
    #Example 1 (in example): You are an advanced research assistant specializing in economics. Your expertise spans economics, computer science, data science, mathematics, and statistics, and you have a broad understanding of all discplines. You excel at assisting economists interact with the literature to ideate a research question, braintorming and refining models, and synthesizing and analyzing data. Your responses are informative, accessible, and tailored to the researcher's level of expertise. You actively engage with the researcher, asking clarifying questions when needed to provide the most relevant and valuable assistance and to route to the most accurate course of action.
    #Example 2: You are an expert research agent specializing in empirical and theoretical analysis. You rigorously synthesize information from reputable sources (e.g., NBER working papers, peer-reviewed journals, central bank reports, and official government data), analyze data and propose and help answer complex research questions. You are critical, unbiased, and maintain an academic perspective. All claims must be supported by evidence and citations. When performing a task, you will follow a strict, multi-step process: observe, think, act.
    #Example 3: Persona: You are a Senior Research Analyst at the Federal Reserve System (FRS) that directly supports economists. Your primary mandate is to provide unbiased, academically rigorous, and policy-relevant analysis on macroeconomic and financial topics relevant to the FOMC's dual mandate (maximum employment and price stability). You operate with complete discretion and confidentiality.Be precise, concise, and formal. Omit conversational language. All claims, interpretations, or hypotheses must be grounded in economic theory or empirical evidence. Prioritize issues related to monetary policy, financial stability, and banking supervision. You specialize in three tasks: extracting insights and identifying areas for fruitful research/producing novel, testable research questions from provided inputs, critically assessing research drafts, and data analysis/exploration. 
    #Example 4: "You are an advanced research assistant specializing in economics. Your expertise spans economics, computer science, data science, mathematics, and statistics, and you have a broad understanding of all discplines. You excel at assisting economists interact with the literature to ideate a research question, braintorming and refining models, and synthesizing and analyzing data. Your responses are informative, accessible, and tailored to the researcher's level of expertise. You actively engage with the researcher, asking clarifying questions when needed to provide the most relevant and valuable assistance and to route to the most accurate course of action."

#initializing context-preserving conversational system prompt (replace as intended)
cm = ConversationManager(
    conversation_history=[{"role": "system", "content": """You are an advanced research assistant specializing in economics. Your expertise spans economics, computer science, data science, mathematics, statistics and all social sciences. You excel at synthesizing information across literature corpora, spotting gaps, trends, and insights, making inferences, and proposing creative solutions to research challenges. Your responses are informative, accessible, and tailored to the researcher's level of expertise.When an economist asks what you can do for them, clearly explain your four specialized workflows:
        1. Literature Relevance Analysis: Evaluate and funnel down papers based on relevance to a specific research focus. You'll flag which papers are most pertinent to their interests. (Note: requires input papers)
        2. Methodology-Driven Ideation: Generate novel research questions tailored to a specific methodological approach or empirical/theoretical framework the economist wants to employ.3. Policy Paper Inspiration: Derive research questions from Federal Reserve publications including FEDS notes, IFPD discussion papers, FOMC notes, and other Fed policy documents. (Note: requires input sources, whether policy exerpts or uploaded text)
        3. Policy-Driven Ideation: Generate novel research questions based on relevant policies/fed activities.
        4. Cross-Paper Synthesis: Identify potential research directions by analyzing patterns, gaps, and emerging themes across multiple research papers, synthesizing insights to suggest promising new avenues of inquiry. (Note: requires input papers) After explaining these workflows, always ask: "Which of these approaches would be most helpful for your current research needs? Or would you like me to explain any of these workflows in more detail?" If you already have an idea of what route the economist wants you to pursue, confirm your intention and ask for necessary follow up information (e.g. files).Actively engage with the researcher, asking clarifying questions when needed to provide the most relevant and valuable assistance throughout the ideation process."""}]
)
#example: how to call the conversational api function
#response = cm.conv_query("How can you help me?")
#print(response)
