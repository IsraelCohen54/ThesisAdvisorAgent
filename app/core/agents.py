# app/core/agents.py
import os
import logging
from typing import Dict, Any

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from app.infrastructure.tools import GoogleScholarTool, PubMedTool

logger = logging.getLogger("ThesisAdvocator")

retry_config = types.HttpRetryOptions(
    attempts=5,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# -------------------------------
# IMPORTANT: explicit wrapper functions so the model must call
# the tool names we control (avoids "execute" hallucination).
# Now returning structured dicts (Dict[str, Any]).
# -------------------------------

def google_scholar_execute(query: str) -> Dict[str, Any]:
    """
    Wrapper named exactly 'google_scholar_execute' (FunctionTool will expose it).
    Returns a dict: {'result': [ {title, source, link, snippet}, ... ] } or {'error': ...}
    """
    return GoogleScholarTool().execute(query)


def pubmed_execute(query: str) -> Dict[str, Any]:
    """
    Wrapper named exactly 'pubmed_execute' (FunctionTool will expose it).
    Returns a dict: {'result': [ {title, authors, source, link, snippet, pmid}, ... ] } or {'error': ...}
    """
    return PubMedTool(max_results=5).execute(query)


# Create FunctionTool objects from wrappers and set explicit .name
_scholar_fn = FunctionTool(func=google_scholar_execute)
_scholar_fn.name = "google_scholar_execute"
_scholar_fn.description = "Search Google Scholar via SerpAPI and return up to 5 structured results (dict {'result': [...]})."

_pubmed_fn = FunctionTool(func=pubmed_execute)
_pubmed_fn.name = "pubmed_execute"
_pubmed_fn.description = "Search PubMed (Entrez) and return up to 5 structured results (dict {'result': [...]})."


# --- Researcher Agents (specialists) ---
ScholarResearcher = LlmAgent(
    name="ScholarResearcher",
    model=Gemini(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"),
                 retry_options=retry_config, temperature=0),
    description="Specialist agent: search Google Scholar and return formatted top results.",
    instruction=(
        "You are ScholarResearcher. CALL the single tool named 'google_scholar_execute' with the "
        "user's thesis as a single string argument and then stop. The tool returns a structured dict "
        "in the form {'result':[...]} — forward that result (do NOT add extraneous commentary)."
    ),
    tools=[_scholar_fn],
)

PubMedResearcher = LlmAgent(
    name="PubMedResearcher",
    model=Gemini(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"),
                 retry_options=retry_config, temperature=0),
    description="Specialist agent: search PubMed and return formatted top results.",
    instruction=(
        "You are PubMedResearcher. CALL the single tool named 'pubmed_execute' with the "
        "user's thesis as a single string argument and then stop. The tool returns a structured dict "
        "in the form {'result':[...]} — forward that result (do NOT add extraneous commentary)."
    ),
    tools=[_pubmed_fn],
)

# Wrap researcher agents as AgentTool so Router/Talk can invoke them
ScholarAgentTool = AgentTool(agent=ScholarResearcher)
ScholarAgentTool.name = "ScholarResearcher"
ScholarAgentTool.description = "Invoke ScholarResearcher to find Google Scholar results."

PubMedAgentTool = AgentTool(agent=PubMedResearcher)
PubMedAgentTool.name = "PubMedResearcher"
PubMedAgentTool.description = "Invoke PubMedResearcher to find PubMed results."

# --- Router Agent (decides which researcher to call) ---
router_instruction = """
You are the ThesisAdvocatorRouter. For each user request YOU MUST CALL exactly one of these tools:
- Call 'PubMedResearcher' if the topic is biomedical, clinical, or life-science related.
- Otherwise call 'ScholarResearcher'.

Use the AgentTool interface (i.e., issue a single function/tool call). Do NOT answer the user directly.
After the researcher returns, it will provide formatted structured results (a dict with key 'result').
"""

RouterAgent = LlmAgent(
    name="ThesisAdvocatorRouter",
    model=Gemini(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"),
                 retry_options=retry_config, temperature=0),
    instruction=router_instruction,
    tools=[PubMedAgentTool, ScholarAgentTool],
)

# Expose Router as an AgentTool (so TalkAgent can call Router)
RouterAgentTool = AgentTool(agent=RouterAgent)
RouterAgentTool.name = "Router"
RouterAgentTool.description = "Invoke the Router to pick and call the appropriate researcher agent."

# --- TalkAgent: user-facing root agent ---
talk_instruction = """
You are TalkAgent, the user-facing conversational agent.
When the user sends a thesis/idea:
  1) CALL the tool named 'Router' exactly once with the thesis string as input. Do NOT answer the user directly.
  2) When the Router returns results (coming from a researcher), forward those results to the human plainly.
  3) If the human replies 'refine' or provides a new thesis, call 'Router' again with the new thesis (fresh search).
Always produce concise assistant messages and let the human decide the next step.
"""

TalkAgent = LlmAgent(
    name="TalkAgent",
    model=Gemini(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"),
                 retry_options=retry_config, temperature=0),
    instruction=talk_instruction,
    tools=[RouterAgentTool],
)


def get_talk_agent():
    return TalkAgent
