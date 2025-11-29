import os
import logging
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from app.infrastructure.tools import GoogleScholarTool, PubMedTool

logger = logging.getLogger("ThesisAdvocator")

# Configure Retry
retry_config = types.HttpRetryOptions(
    attempts=5,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# --- Tool Wrappers ---

def google_scholar_execute(query: str) -> str:
    """Search Google Scholar. Returns a readable string of results."""
    # The Logic handles formatting internally in the tool class or here
    # We ensure we return a string to avoid ADK parsing errors
    try:
        resp = GoogleScholarTool().execute(query)
        # If it returns a dict, we cast to string so the Agent can read it
        return str(resp)
    except Exception as e:
        return f"Error running Google Scholar: {e}"


def pubmed_execute(query: str) -> str:
    """Search PubMed. Returns a readable string of results."""
    try:
        resp = PubMedTool(max_results=5).execute(query)
        return str(resp)
    except Exception as e:
        return f"Error running PubMed: {e}"


# --- Create FunctionTools ---
# We give these DIRECTLY to the TalkAgent. No "Router" needed.
_scholar_fn = FunctionTool(func=google_scholar_execute)
_scholar_fn.name = "google_scholar_execute"
_scholar_fn.description = "Useful for queries about history, business, social science, or general topics."

_pubmed_fn = FunctionTool(func=pubmed_execute)
_pubmed_fn.name = "pubmed_execute"
_pubmed_fn.description = "Useful ONLY for queries about biology, medicine, clinical trials, diseases, or health."

# --- The Talk Agent (Root) ---
talk_instruction = """
You are the Thesis Advocator. Your job is to find academic sources for the user's thesis idea.

RULES:
1. Analyze the user's thesis.
2. Call EXACTLY ONE tool:
   - Call 'pubmed_execute' if the topic is biological/medical.
   - Call 'google_scholar_execute' for everything else.
3. OUTPUT: Pass the user's query to the tool.
4. When the tool returns results, forward them EXACTLY as is. Do not summarize. Do not add intro/outro text. Just output the tool result.
"""

TalkAgent = LlmAgent(
    name="TalkAgent",
    model=Gemini(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"),
                 retry_options=retry_config, temperature=0),
    instruction=talk_instruction,
    # Give the agent both tools directly
    tools=[_scholar_fn, _pubmed_fn],
)


def get_talk_agent():
    return TalkAgent
