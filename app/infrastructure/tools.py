# app/infrastructure/tools.py
import logging
import os
import requests
from typing import Any, Dict

logger = logging.getLogger("ThesisAdvocator")


class GoogleScholarTool:
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
            "num": 5  # Request 5 results
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
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

            # Return a formatted string if results found, else message
            final_text = "\n".join(results) if results else "No academic results found."
            return {"result": final_text}

        except requests.exceptions.RequestException as e:
            logger.error(f"SerpApi request failed: {e}")
            return {"error": f"SerpApi Error: {e}"}