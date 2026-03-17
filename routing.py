#Will contain the routing class that directs an economist to a workflow given that they arrive at the evaluation agent.
import os
import sys
import json
import re
from typing import List, Dict, Optional

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))      # for eval/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # for architecture.py

from architecture import ConversationManager, query_api_single
from eval.workflows.referee import WorkflowReferee
from eval.workflows.section_eval import WorkflowFeedback

class RoutingAgent:
    def __init__(self, llm: ConversationManager):
        self.llm = llm
        self.workflow_map = {
            "referee": WorkflowReferee(llm),
            "feedback": WorkflowFeedback(llm),
        }

    def is_query_clear(self, user_prompt: str) -> bool:
        """Use LLM to decide if the query is too vague or ambiguous."""
        check_prompt = f"""
        The user wrote: "{user_prompt}"

        Is this a clear, specific, and actionable question that you can assign to a workflow? 

        Respond only with: YES or NO
        """
        response = query_api_single(check_prompt).strip().lower()
        return "yes" in response

    def ask_for_clarification(self, user_prompt: str) -> str:
        """Ask a follow-up question to clarify the user intent."""
        clarifying_prompt = f"""The user wrote: "{user_prompt}". This is vague or under-specified. What one concise, clarifying question would you ask the user to better understand their research intent to discriminate between workflows? Respond only with the question (no quotes or explanations)."""
        return query_api_single(clarifying_prompt).strip()

    def classify_prompt(self, user_prompt: str) -> str:
        """Route the prompt to the right workflow using an LLM classifier."""
        routing_prompt = f"""
        You are routing a user prompt to one of the following workflows:
        - feedback: goes section by section through a research paper to suggest changes/improvements
        - referee: compares a referee report to a paper holistically, then creates a revised draft section by section 

        Prompt: "{user_prompt}"

        Which workflow does this best match? Respond only with one of:
        feedback, referee
        """
        category = query_api_single(routing_prompt).strip().lower()
        if category not in self.workflow_map:
            category = "feedback"  # fallback
        return category

    def run(self, user_prompt: str, pdf_paths: List[str]) -> Dict[str, any]:
        """Main control loop."""
        history = {}

        if not self.is_query_clear(user_prompt):
            question = self.ask_for_clarification(user_prompt)
            print(f"\n? Clarifying question: {question}")
            user_prompt = input("Your answer: ").strip()
            history["clarification"] = question
            history["user_followup"] = user_prompt

        category = self.classify_prompt(user_prompt)
        history["chosen_workflow"] = category

        print(f"\n? Routing to: {category.upper()} workflow...\n")

        workflow = self.workflow_map[category]
        result = workflow.run(pdf_paths, research_agenda=user_prompt)

        return {
            "workflow": category,
            "results": result,
            "history": history
        }

if __name__ == "__main__":
    from architecture import ConversationManager

    llm = ConversationManager(conversation_history =  [])
    agent = RoutingAgent(llm)

    query = "I have feedback from a journal that I have submitted a research paper to and want according revisions."

    # ? Example PDF paths
    pdfs = [
        "ideation/papers/FEDS_note2"
    ]

    results = agent.run(query, pdfs)

    # Pretty-print results
    import json
    print("\n? Final Output:\n")
    print(json.dumps(results, indent=2))