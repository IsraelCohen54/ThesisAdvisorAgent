import os
import logging
from google.adk.agents import LlmAgent, Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools.function_tool import FunctionTool
from app.interfaces.interfaces import AgentInterface
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from app.infrastructure.tools import GoogleScholarTool  # Only GoogleScholarTool remains
from google.genai import types

logger = logging.getLogger("ThesisAdvocator")

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


class ThesisAdvocatorAgent(Agent, AgentInterface):
    """
    The main routing agent. Classifies user thesis and delegates search to
    the specialized PubMed or Google Search tools (A2A principle).
    """

    def __init__(self, session_service: any):  # Session service for memory (Next step)
        # Initialize the underlying ADK LlmAgent/Agent
        # We use 'Agent' as a fallback if 'LlmAgent' is not the correct name.
        agent_class = LlmAgent if 'LlmAgent' in globals() else Agent
        model = Gemini(model="gemini-2.5-flash-lite",  # "gemini-2.0-flash-exp", # => billing model
                       api_key=os.getenv("GOOGLE_API_KEY"),
                       retry_config=retry_config)

        # Call the Super Constructor (BaseAgent/Agent)
        super().__init__(
            name="ThesisAdvocatorRouter",
            description="Routes academic debate topics to specialized search tools.",
            tools=[self.google_scholar_tool, self.pubmed_tool]
        )

        self.google_scholar_tool = FunctionTool(func=GoogleScholarTool().execute)
        self.google_scholar_tool.name = "google_scholar_tool"
        self.google_scholar_tool.description = "Searches high-quality academic papers, citations, and university data."

        self.pubmed_tool = McpToolset(
            connection_params=StdioConnectionParams(
                # The command to run the external Node.js server
                server_params=StdioServerParameters(
                    # Assuming you have Node.js/npm installed, this runs the package
                    command="npx",
                    args=[
                        "-y",  # Auto-confirm install
                        "@cyanheads/pubmed-mcp-server",
                    ],
                    # Optional: Configure your NCBI Key (Highly recommended to avoid rate limits)
                    # You would need to pass environment variables here, e.g.,
                    # env={"NCBI_API_KEY": os.getenv("NCBI_API_KEY") or ""}
                    tool_filter=["pubmed_search"],  # todo You may need to verify the exact name
                ),
                timeout=60,  # Allow a generous timeout for the first start
            )
        )

        self.pubmed_tool.name = "pubmed_mcp_tool"
        self.pubmed_tool.description = "Searches for biomedical, health, and life science research papers (via MCP)."

        # System instruction for the Router
        system_prompt = """You are the Thesis Advocator Agent. Your task is to analyze the user's thesis. 
            If the thesis is primarily about medicine, biology, or health sciences, use the 'pubmed_mcp_tool'. 
            Otherwise, use the 'google_scholar_tool' for all general academic research. 
            Do not answer the question directly; only decide which tool to call."""

        self.adk_agent = agent_class(
            name="ThesisAdvocatorRouter",
            model=model,
            # tools=[FunctionTool(func=GoogleScholarTool().execute), self.pubmed_tool],
            tools=[self.google_scholar_tool, self.pubmed_tool],
            instruction=system_prompt,
        )

    def run(self, input_data: str, session_id: str) -> str:
        logger.info(f"[Router] Received thesis: {input_data}")
        # Note: In ADK, calling the agent's run method handles tool selection automatically.
        response = self.adk_agent.run(prompt=input_data)

        # We process the final response to see if a tool call was made
        if response.tool_calls:
            # If a tool was called, the ADK handles execution and returns the result.
            logger.info(f"[Router] Delegated to: {response.tool_calls[0].function.name}")
            return "Tool Execution Triggered successfully."
        else:
            return response.text  # Should be a very rare case if prompt is followed
