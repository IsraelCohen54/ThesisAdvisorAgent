# app/core/debater_agent.py
import logging
import os
from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini

logger = logging.getLogger("DebaterAgent")

# todo ~~~~~~~~~~~~~~ currently as place holder ~~~
class DebaterAgent(Agent):
    """
    The Debater Agent: Receives search results and the final thesis,
    then performs analysis. This is the A2A Target Agent.
    """

    def __init__(self):
        model = Gemini(model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

        system_prompt = (
            "You are a sophisticated Academic Debater. Your task is to analyze "
            "the user's confirmed thesis and the relevant academic papers provided. "
            "Generate a strong introductory paragraph outlining potential areas of disagreement "
            "or key arguments based on the search results. Respond concisely."
        )

        super().__init__(
            name="ThesisDebaterAgent",
            model=model,
            instruction=system_prompt,
            # No tools needed for this agent
            tools=[],
        )

    # We will let the base Agent.run handle the LLM call in this case.
    # The input will be the search results string + user thesis.