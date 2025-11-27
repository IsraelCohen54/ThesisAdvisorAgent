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

    def execute(self, query: str) -> Dict[str, Any]:
        """Return {'result': [ {title, source, authors, link, snippet, pmid}, ... ] }"""
        term = self._make_search_term(query)
        logger.info(f"[PubMedTool] Searching: {term}")

        try:
            with Entrez.esearch(db="pubmed", term=term, retmax=self.max_results, sort="relevance") as handle:
                record = Entrez.read(handle)
            pmids = record.get("IdList", [])

            if not pmids:
                return {"result": []}

            id_csv = ",".join(pmids)
            with Entrez.efetch(db="pubmed", id=id_csv, rettype="medline", retmode="text") as handle:
                records = list(Medline.parse(handle))

            out = []
            for rec, pmid in zip(records, pmids):
                title = rec.get("TI", "No Title")
                authors_list = rec.get("AU", []) or []
                authors = ", ".join(authors_list[:3])  # keep it short
                source = f"{rec.get('TA', '')} ({rec.get('DP', '')})".strip()
                snippet = (rec.get("AB") or "")[:300]
                link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                out.append({
                    "title": title,
                    "authors": authors,
                    "source": source,
                    "link": link,
                    "snippet": snippet,
                    "pmid": pmid
                })
            return {"result": out}

        except Exception as e:
            logger.exception("PubMed search failed")
            return {"error": str(e)}


class GoogleScholarTool:
    """Google Scholar search using SerpApi. Returns formatted string for LLM consumption."""

    def execute(self, query: str) -> Dict[str, Any]:
        """Return {'result': [ {title, source, link, snippet}, ... ] }"""
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            logger.error("SERPAPI_API_KEY not found. Skipping Google Scholar search.")
            return {"error": "API Key Missing"}

        logger.info(f"[ScholarTool] Searching: {query}")

        try:
            params = {"engine": "google_scholar", "q": query, "api_key": api_key, "num": 5}
            response = requests.get("https://serpapi.com/search.json", params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for r in data.get("organic_results", [])[:5]:
                title = r.get("title", "No Title")
                snippet = r.get("snippet", "") or r.get("snippet_highlighted", "")
                link = r.get("link", "N/A")
                pub_info = ""
                # SerpApi sometimes exposes publication detail in different keys:
                if r.get("publication_info"):
                    pub_info = r["publication_info"].get("summary", "") or r["publication_info"].get("metadata", "")
                results.append({
                    "title": title,
                    "source": pub_info,
                    "link": link,
                    "snippet": (snippet or "")[:300]
                })
            return {"result": results}

        except requests.exceptions.RequestException as e:
            logger.error(f"SerpApi request failed: {e}")
            return {"error": f"SerpApi Error: {e}"}
