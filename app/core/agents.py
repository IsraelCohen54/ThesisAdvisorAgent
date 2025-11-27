import os
import logging
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from app.infrastructure.tools import GoogleScholarTool, PubMedTool

logger = logging.getLogger("ThesisAdvocator")

retry_config = types.HttpRetryOptions(
    attempts=5,
    initial_delay=2,
    http_status_codes=[429, 500, 503, 504],
)


class ThesisAdvocatorAgent(Agent):
    """
    The Router Agent.
    """

    def __init__(self):
        # 1. Setup Model
        model = Gemini(
            model="gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            retry_options=retry_config,
            temperature=0
        )

        # 2. Setup Tools with Explicit Wrappers
        # We define local functions to force the name the LLM sees.

        # --- Wrapper for Scholar ---
        scholar_instance = GoogleScholarTool()

        def google_scholar_tool(query: str):
            """Searches most similar high-quality preferably academic papers/thesis/articles via Google Scholar.
            Use for general, history, or humanities topics."""
            return scholar_instance.execute(query)

        # Force the name just to be safe (though 'def' usually handles it)
        google_scholar_tool.__name__ = "google_scholar_tool"

        # --- Wrapper for PubMed ---
        pubmed_instance = PubMedTool(max_results=5)

        def pubmed_tool(query: str):
            """Searches most similar biomedical literature via PubMed.
            Use for biology, medicine, and health topics."""
            return pubmed_instance.execute(query)

        # Force the name
        pubmed_tool.__name__ = "pubmed_tool"

        # Create the ADK Tools using the WRAPPERS, not the class methods
        scholar_tool_obj = FunctionTool(func=google_scholar_tool)
        scholar_tool_obj.name = "google_scholar_tool"  # <- explicit name the model should call
        scholar_tool_obj.description = "google_scholar_tool(query: str) -> returns formatted scholar results."

        pubmed_tool_obj = FunctionTool(func=pubmed_tool)
        pubmed_tool_obj.name = "pubmed_tool"  # <- explicit name the model should call
        pubmed_tool_obj.description = "pubmed_tool(query: str) -> returns formatted pubmed results."

        # 3. System Instruction
        system_prompt = """You are the Thesis Advocator Router.
        Your job: choose the single best specialist tool and CALL it immediately (do not answer directly).
        IMPORTANT: You MUST call exactly one of the available tool functions by name and not output a normal-text answer instead of calling a tool.

        Rules:
        - If the thesis is biomedical/medical/clinical in topic, CALL the function named: "pubmed_tool"
        - Otherwise, CALL the function named: "google_scholar_tool"
        - Do NOT output extra natural language instead of calling the tool.
        - After the tool returns, summarize the 5 most similar matches for the user in this format:

        Title:
        Summary:
        Link:

        Keep summaries concise and factual.
        """

        # 4. Initialize Base Agent
        super().__init__(
            name="ThesisAdvocatorRouter",
            model=model,
            tools=[scholar_tool_obj, pubmed_tool_obj],  # <<-- use the FunctionTool objs
            instruction=system_prompt,
        )
        logger.info("Registered tools: %s", [t.name for t in self.tools])


# Factory function is now much simpler (or can be removed if you call class directly)
def get_router_agent():
    return ThesisAdvocatorAgent()
