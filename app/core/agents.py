import logging
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from app.infrastructure.tools import GoogleScholarTool, PubMedTool

logger = logging.getLogger("ThesisAdvisor")

# Configure Retry
retry_config = types.HttpRetryOptions(
    attempts=5,
    initial_delay=2,
    http_status_codes=[429, 500, 503, 504],
    max_delay=2,
    exp_base=1.5
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
    """Search PubMed. If PubMed returns no results or an error, fall back to Google Scholar."""
    try:
        resp = PubMedTool(max_results=5).execute(query)
    except Exception as e:
        # PubMed client crashed — try scholar fallback
        logger.warning("[pubmed_execute] PubMed call raised exception, falling back to Google Scholar: %s", e)
        try: # sresp = scholar response
            sresp = GoogleScholarTool().execute(query)
            return str(sresp)
        except Exception as e2:
            return f"Error running PubMed and Scholar fallback: {e2}"

    # If PubMed returned an explicit error or an empty 'result' list -> fallback to Scholar
    try:
        if isinstance(resp, dict):
            if resp.get("error"):
                logger.info("[pubmed_execute] PubMed returned error, falling back to Google Scholar")
                sresp = GoogleScholarTool().execute(query)
                return str(sresp)
            result = resp.get("result", None)
            if isinstance(result, (list, tuple)) and len(result) == 0:
                logger.info("[pubmed_execute] PubMed returned no results, falling back to Google Scholar")
                sresp = GoogleScholarTool().execute(query)
                return str(sresp)
    except Exception as e:
        logger.debug("[pubmed_execute] Unexpected parsing error; attempting Scholar fallback: %s", e)
        try:
            sresp = GoogleScholarTool().execute(query)
            return str(sresp)
        except Exception as e2:
            return f"Error running Scholar fallback after PubMed parse error: {e2}"

    # Normal case: return PubMed result as string (agent expects a string)
    return str(resp)


# --- Create FunctionTools ---
_scholar_fn = FunctionTool(func=google_scholar_execute)
_scholar_fn.name = "google_scholar_execute"
_scholar_fn.description = "Useful for queries about history, business, social science, or general topics."

_pubmed_fn = FunctionTool(func=pubmed_execute)
_pubmed_fn.name = "pubmed_execute"
_pubmed_fn.description = "Useful ONLY for queries about biology, medicine, clinical trials, diseases, or health."

# --- The Talk Agent (Root) ---
talk_instruction = """
You are the Thesis Advisor. Your job is to find academic sources for the user's thesis idea.

RULES:
1. Analyze the user's thesis.
2. Call EXACTLY ONE tool:
   - Call 'pubmed_execute' if the topic is biological/medical.
   - Call 'google_scholar_execute' for everything else.
3. OUTPUT: Pass the user's query to the tool.
4. When the tool returns results, forward them EXACTLY as is. Do not summarize. Do not add intro/outro text. Just output the tool result.
"""

DialogAgent1 = LlmAgent(
    name="DialogAgent1",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction=talk_instruction,
    # Give the agent both tools directly
    tools=[_scholar_fn, _pubmed_fn],

)


def get_dialog_agent1():
    return DialogAgent1
