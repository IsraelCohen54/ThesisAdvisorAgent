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
            model="gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            retry_options=retry_config
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
        pubmed_tool_obj = FunctionTool(func=pubmed_tool)

        # 3. System Instruction
        system_prompt = """You are the Thesis Advocator Router. 
        Your task is to choose most appropriate tool to search for the most similar thesis or academic articles by title

        1. Analyze the thesis topic.
        2. If it is biomedical or health related, call 'pubmed_tool_obj'.
        3. Otherwise, call 'scholar_tool_obj'.
        DO NOT ask for clarification. Pick the best tool and run it immediately.
        4. After you call the tool, you MUST return the tool's 5 most similar match by title directly to the user. 
        5. process, summarize, and reformat the results to be representable in this way:
        Title:
        Summary:
        Link:
        Make it simple but concise and informative.
        """

        # 4. Initialize Base Agent
        super().__init__(
            name="ThesisAdvocatorRouter",
            model=model,
            tools=[scholar_tool_obj, pubmed_tool_obj],
            instruction=system_prompt,
        )


# Factory function is now much simpler (or can be removed if you call class directly)
def get_router_agent():
    return ThesisAdvocatorAgent()
