import logging
import os
import requests
from typing import Any, Dict, List
from Bio import Entrez, Medline  # pip install biopython

logger = logging.getLogger("ThesisAdvocator")

# Configure Entrez
Entrez.email = os.getenv("NCBI_CONTACT_EMAIL", "your-email@example.com")
api_key = os.getenv("NCBI_API_KEY")
if api_key:
    Entrez.api_key = api_key


class PubMedTool:
    """PubMed search tool using Biopython. Returns formatted string for LLM consumption."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def _make_search_term(self, query: str) -> str:
        q = query.strip()
        # Heuristic: short queries -> restrict to Title/Abstract for precision
        if len(q.split()) <= 5:
            return f"{q}[Title/Abstract]"
        return q

    def execute(self, query: str) -> str:
        """Executes search and returns a pre-formatted string for the Agent."""
        term = self._make_search_term(query)
        logger.info(f"[PubMedTool] Searching: {term}")

        try:
            # 1. Search
            with Entrez.esearch(db="pubmed", term=term, retmax=self.max_results, sort="relevance") as handle:
                record = Entrez.read(handle)
            pmids = record.get("IdList", [])

            if not pmids:
                return "No PubMed results found."

            # 2. Fetch Details
            id_csv = ",".join(pmids)
            with Entrez.efetch(db="pubmed", id=id_csv, rettype="medline", retmode="text") as handle:
                records = list(Medline.parse(handle))

            # 3. Format
            output_lines = ["--- PUBMED SEARCH RESULTS ---"]
            for rec in records:
                title = rec.get("TI", "No Title")
                authors = ", ".join(rec.get("AU", [])[:3])  # Limit authors for brevity
                source = f"{rec.get('TA', '')} ({rec.get('DP', '')})"
                pmid = rec.get("PMID", "")
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                snippet = rec.get("AB", "")[:400] + "..." if rec.get("AB") else "No abstract."

                output_lines.append(f"TITLE: {title}")
                output_lines.append(f"SOURCE: {source}")
                output_lines.append(f"AUTHORS: {authors}")
                output_lines.append(f"LINK: {url}")
                output_lines.append(f"SNIPPET: {snippet}\n")

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"PubMed Error: {e}")
            return f"Error searching PubMed: {str(e)}"


class GoogleScholarTool:
    """Google Scholar search using SerpApi. Returns formatted string for LLM consumption."""

    def execute(self, query: str) -> str:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "Error: SERPAPI_API_KEY not found."

        logger.info(f"[ScholarTool] Searching: {query}")

        try:
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": api_key,
                "num": 5
            }
            response = requests.get("https://serpapi.com/search.json", params=params)
            response.raise_for_status()
            data = response.json()

            results = data.get("organic_results", [])
            if not results:
                return "No Google Scholar results found."

            output_lines = ["--- GOOGLE SCHOLAR RESULTS ---"]
            for r in results:
                title = r.get("title", "No Title")
                snippet = r.get("snippet", "No snippet")
                link = r.get("link", "N/A")
                pub_info = r.get("publication_info", {}).get("summary", "")

                output_lines.append(f"TITLE: {title}")
                output_lines.append(f"SOURCE: {pub_info}")
                output_lines.append(f"LINK: {link}")
                output_lines.append(f"SNIPPET: {snippet}\n")

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"SerpApi Error: {e}")
            return f"Error searching Scholar: {str(e)}"