# Abstract Base Classes
import logging
import os
import requests  # New import
from app.interfaces.interfaces import ToolInterface
from typing import Any, Dict, List

logger = logging.getLogger("ThesisAdvocator")


# We are replacing the old GoogleSearchTool with this one
class GoogleScholarTool(ToolInterface):
    """Concrete implementation for academic search using SerpApi."""

    def execute(self, query: str) -> Dict[str, Any]:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            logger.error("SERPAPI_API_KEY not found. Skipping Google Scholar search.")
            return {"error": "API Key Missing"}

        logger.info(f"[Tool] Executing Google Scholar Search via SerpApi for: {query}")

        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": api_key,
            "num": 5  # Request 5 results as planned
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses
            data = response.json()

            # Format the organic results for the agent
            results = []
            for result in data.get("organic_results", []):
                results.append(
                    f"TITLE: {result.get('title')}\n"
                    f"SNIPPET: {result.get('snippet', 'No snippet available.')}\n"
                    f"SOURCE: {result.get('publication_detail', 'N/A')}\n"
                    f"LINK: {result.get('link', 'N/A')}\n"
                    "---"
                )

            return {"result": "\n".join(results)}

        except requests.exceptions.RequestException as e:
            logger.error(f"SerpApi request failed: {e}")
            return {"error": f"SerpApi Error: {e}"}


# PubMed Agent using the Model Context Protocol (MCP) (Still a placeholder for now)
class PubMedMCPTool(ToolInterface):
    """Placeholder tool for the MCP Server interaction."""

    def execute(self, query: str) -> Dict[str, Any]:
        logger.info(f"[Tool] PubMed/MCP activated for: {query}")
        # Note: We will handle the MCP client integration in a later step.
        return {"status": "AWAITING_MCP_SERVER", "query": query}
