# agents.py
import os
import logging
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.genai import types

from app.infrastructure.tools import GoogleScholarTool

logger = logging.getLogger("ThesisAdvocator")

retry_config = types.HttpRetryOptions(
    attempts=5,
    http_status_codes=[429, 500, 503, 504],
)


# ✅ FIX: Inherit ONLY from Agent. We dropped AgentInterface to stop Pydantic errors.
class ThesisAdvocatorAgent(Agent):
    """
    The main routing agent. Classifies user thesis and delegates search.
    """

    def __init__(self):
        # 1. Setup Model
        model = Gemini(
            model="gemini-2.0-flash-exp",
            api_key=os.getenv("GEMINI_API_KEY"),
            retry_options=retry_config
        )

        # 2. Setup Tools LOCALLY (Do not bind to self yet)

        # Google Scholar
        scholar_tool = FunctionTool(func=GoogleScholarTool().execute)
        scholar_tool.name = "google_scholar_tool"
        scholar_tool.description = "Searches high-quality academic papers, citations, and university data."

        # PubMed MCP
        pubmed_tool = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="npx",
                    args=["-y", "@cyanheads/pubmed-mcp-server"],
                    tool_filter=["pubmed_search"],
                ),
                timeout=60,
            )
        )
        pubmed_tool.name = "pubmed_mcp_tool"
        pubmed_tool.description = "Searches for biomedical, health, and life science research papers (via MCP)."

        # System Instruction
        system_prompt = """You are the Thesis Advocator Agent. Your task is to analyze the user's thesis. 
            If the thesis is primarily about medicine, biology, or health sciences, use the 'pubmed_mcp_tool'. 
            Otherwise, use the 'google_scholar_tool' for all general academic research. 
            Do not answer the question directly; only decide which tool to call."""

        # 3. Initialize Parent (The actual Agent)
        super().__init__(
            name="ThesisAdvocatorRouter",
            model=model,
            tools=[scholar_tool, pubmed_tool],  # Pass local vars here!
            instruction=system_prompt,
        )