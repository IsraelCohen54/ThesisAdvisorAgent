#!/usr/bin/env python3
# agent_evaluation.py
import os
import uuid
import json
import ast
import asyncio
import logging
import re
from difflib import SequenceMatcher
from types import SimpleNamespace
from typing import Any, List, Tuple, Optional, Dict

from google.genai import types as gen_types
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.google_llm import Gemini

from app.core.agents import get_talk_agent

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AgentEval")

# -------------------------
# Utility: robust parse & extract titles (from earlier helper)
# -------------------------
TITLE_KEYS = ("title", "paper_title", "headline", "TI", "name")
FALLBACK_SNIPPET_KEYS = ("snippet", "abstract", "summary", "AB")


def _safe_parse(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    if not s:
        return raw
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return raw


def _normalize_for_compare(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _collect_title_candidates(obj: Any) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []

    def walk(o):
        if o is None:
            return
        if isinstance(o, dict):
            for k in TITLE_KEYS:
                if k in o and isinstance(o[k], str) and o[k].strip():
                    out.append((o[k].strip(), o))
            for k in FALLBACK_SNIPPET_KEYS:
                if k in o and isinstance(o[k], str) and o[k].strip():
                    out.append((o[k].strip(), o))
            for k in ("result", "results", "organic_results", "items", "articles", "papers"):
                if k in o:
                    walk(o[k])
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for it in o:
                walk(it)
        else:
            if isinstance(o, str) and o.strip():
                out.append((o.strip(), o))

    walk(obj)
    # dedupe preserving first appearance
    seen = set()
    deduped = []
    for title, origin in out:
        key = " ".join(title.split())
        if key not in seen:
            seen.add(key)
            deduped.append((title, origin))
    return deduped


def compare_target_to_response(target_title: str, raw_response: Any, fuzz_threshold_exact: float = 0.98) -> Dict[
    str, Any]:
    """
    Compare a target title to the agent/tool response.

    Returns a dict:
      {
        "exact_found": bool,
        "best_ratio": float (0..1),
        "best_candidate_title": str|None,
        "best_candidate": object|None,
        "all_candidates": [titles...]
      }
    """
    parsed = _safe_parse(raw_response)

    # collect structured title candidates
    candidates = _collect_title_candidates(parsed)

    # If no structured candidates found, try to interpret parsed as a simple string candidate
    if not candidates:
        if isinstance(parsed, str) and parsed.strip():
            candidates = [(parsed.strip(), parsed)]
        else:
            candidates = []

    target_norm = _normalize_for_compare(target_title)
    best_ratio = 0.0
    best_candidate = None
    best_title = None
    exact_found = False

    for cand_title, origin in candidates:
        cand_norm = _normalize_for_compare(cand_title)
        if cand_norm == target_norm:
            # exact normalized match
            exact_found = True
            best_ratio = 1.0
            best_candidate = origin
            best_title = cand_title
            break
        ratio = SequenceMatcher(None, target_norm, cand_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_candidate = origin
            best_title = cand_title

    # FALLBACK 1: If no exact normalized match found in structured candidates,
    # check raw stringified response for a normalized substring match (handles Python-dict-as-string)
    if not exact_found:
        try:
            raw_text = str(raw_response or "")
            raw_norm = _normalize_for_compare(raw_text)
            if target_norm and target_norm in raw_norm:
                # exact substring match found in the raw text (normalized)
                exact_found = True
                best_ratio = 1.0
                # attempt to extract the original snippet (unenormalized) containing the title
                # simple approach: find the original target (case-insensitive) inside the raw text
                idx = None
                try:
                    idx = raw_text.lower().index(target_title.lower())
                except ValueError:
                    # try looser: find first occurrence of the first 8 chars normalized
                    small = target_norm.split()[:8]
                    if small:
                        probe = " ".join(small)
                        try:
                            idx = raw_text.lower().index(probe)
                        except ValueError:
                            idx = None
                if idx is not None:
                    # extract a slice around the found index (safety)
                    slice_text = raw_text[idx: idx + max(len(target_title), 200)]
                    best_title = slice_text.strip()
                    best_candidate = raw_response
                else:
                    # fallback to target_title as the best_title
                    best_title = target_title
                    best_candidate = raw_response
        except Exception:
            pass

    # FALLBACK 2: if still nothing, keep returning best fuzzy ratio found
    all_titles = [t for t, _ in _collect_title_candidates(parsed)] if parsed is not None else []

    return {
        "exact_found": exact_found,
        "best_ratio": best_ratio,
        "best_candidate_title": best_title,
        "best_candidate": best_candidate,
        "all_candidates": all_titles[:8],
    }


# -------------------------
# ADK run helpers
# -------------------------
async def run_agent_query(talk_agent, query: str, user_id: str, session_id: str, timeout: int = 60):
    """
    Run the talk agent (async runner.run_async) and collect events.
    Returns: dict(tool_used, tool_output_raw, all_text_parts, raw_events)
    """
    app = App(name="EvalApp", root_agent=talk_agent)
    runner = Runner(app=app, session_service=InMemorySessionService())
    # create session
    await runner.session_service.create_session(app_name="EvalApp", user_id=user_id, session_id=session_id)

    content = gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=query)])
    tool_used = None
    tool_output = None
    text_parts: List[str] = []
    raw_events = []

    try:
        # iterate runner.run_async
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            raw_events.append(event)
            # inspect event content parts
            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                for p in event.content.parts:
                    # if model asked to call a function (function_call)
                    fc = getattr(p, "function_call", None)
                    if fc and getattr(fc, "name", None):
                        tool_used = tool_used or fc.name
                    # if tool returned a function_response
                    fr = getattr(p, "function_response", None)
                    if fr:
                        # function_response.response may be structured
                        resp = getattr(fr, "response", None)
                        if resp is not None:
                            tool_output = resp
                        # name may indicate tool
                        if getattr(fr, "name", None):
                            tool_used = fr.name or tool_used
                    # text
                    if getattr(p, "text", None):
                        text_parts.append(p.text)
            # some events include 'candidates'
            candidates = getattr(event, "candidates", None)
            if candidates:
                try:
                    for cand in candidates:
                        if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                            for p in cand.content.parts:
                                if getattr(p, "text", None):
                                    text_parts.append(p.text)
                                if getattr(p, "function_call", None) and getattr(p.function_call, "name", None):
                                    tool_used = tool_used or p.function_call.name
                                if getattr(p, "function_response", None):
                                    resp = getattr(p.function_response, "response", None)
                                    if resp is not None:
                                        tool_output = resp
                except Exception:
                    pass
    except Exception as e:
        # return partial results on error
        logger.warning("Agent run exception: %s", e)

    # fallback: if the agent printed the tool output as text rather than function_response
    if tool_output is None and text_parts:
        # join and try to parse
        combined = "\n".join(text_parts)
        # try to parse JSON-like list/dict
        parsed = _safe_parse(combined)
        # if parsed looks structured, use it; else keep combined text
        if isinstance(parsed, (dict, list)):
            tool_output = parsed
        else:
            tool_output = combined

    return {
        "tool_used": tool_used,
        "tool_output": tool_output,
        "text": "\n".join(text_parts).strip(),
        "raw_events": raw_events,
    }


# -------------------------
# Gemini similarity evaluator (1..10). fallback -> fuzzy scale
# -------------------------
def gemini_similarity_score_sync(target: str, candidate_text: str, api_key: Optional[str]) -> float:
    """
    Synchronously call Gemini generate_content to request rating 1..10.
    Returns float 1..10. On failure returns fuzzy ratio*10.
    """
    try:
        model = Gemini(model="gemini-2.5-flash", api_key=api_key)
        prompt = (
            "Rate how closely the candidate search result matches the target article title on a scale 1..10.\n\n"
            f"Target title:\n{target}\n\nCandidate (title/snippet):\n{candidate_text}\n\n"
            "Output only a single number between 1 and 10 (inclusive). If you think it's an exact match, return 10."
        )
        resp = model.api_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = getattr(resp, "text", None) or str(resp)
        # extract first integer 1..10
        m = re.search(r"\b([1-9]|10)\b", text)
        if m:
            return float(int(m.group(1)))
        # maybe model returned "8.5" etc
        m2 = re.search(r"\b([0-9](?:\.[0-9])?)\b", text)
        if m2:
            val = float(m2.group(1))
            if 1.0 <= val <= 10.0:
                return val
    except Exception as e:
        logger.debug("Gemini scoring failed: %s", e)

    # fallback: fuzzy ratio scaled
    targ_n = _normalize_for_compare(target)
    cand_n = _normalize_for_compare(candidate_text)
    ratio = SequenceMatcher(None, targ_n, cand_n).ratio()
    return ratio * 10.0


# -------------------------
# Tests specification
# -------------------------
TESTS = [
    # 1 exact PubMed
    {
        "name": "Flagellin exact (PubMed)",
        "query": "Cloning and characterization of flagellin genes and identification of flagellin glycosylation from thermophilic Bacillus species",
        "expected_tool": "pubmed_execute",
        "target_title": "Cloning and characterization of flagellin genes and identification of flagellin glycosylation from thermophilic Bacillus species",
    },
    # 1 variant (pubmed) - not exact wording
    {
        "name": "Flagellin fuzzy (PubMed)",
        "query": "flagellin glycosylation in thermophilic Bacillus species cloning characterization",
        "expected_tool": "pubmed_execute",
        "target_title": "Cloning and characterization of flagellin genes and identification of flagellin glycosylation from thermophilic Bacillus species",
    },
    # 2 exact Google Scholar (crypto)
    {
        "name": "Crypto exact (Scholar)",
        "query": "Unveiling Cryptocurrency Impact on Financial Markets and Traditional Banking Systems: Lessons for Sustainable Blockchain and Interdisciplinary Collaborations",
        "expected_tool": "google_scholar_execute",
        "target_title": "Unveiling Cryptocurrency Impact on Financial Markets and Traditional Banking Systems: Lessons for Sustainable Blockchain and Interdisciplinary Collaborations",
    },
    # 2 variant (Scholar) - fuzzy
    {
        "name": "Crypto fuzzy (Scholar)",
        "query": "cryptocurrency impact on financial markets and banks sustainable blockchain interdisciplinary collaborations study",
        "expected_tool": "google_scholar_execute",
        "target_title": "Unveiling Cryptocurrency Impact on Financial Markets and Traditional Banking Systems: Lessons for Sustainable Blockchain and Interdisciplinary Collaborations",
    },
    # 4 general
    {
        "name": "Sport health general",
        "query": "how doing sport help with your health?",
        "expected_tool": "google_scholar_execute",
        "target_title": None,
    },
]


# -------------------------
# Runner: perform all tests
# -------------------------
async def evaluate_all():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        logger.error("GEMINI_API_KEY not found in environment. Set it and re-run.")
        return

    talk_agent = get_talk_agent()
    user_id = "evaluator_user"
    session_base = uuid.uuid4().hex[:6]

    for i, test in enumerate(TESTS, 1):
        name = test["name"]
        query = test["query"]
        expected_tool = test["expected_tool"]
        target_title = test["target_title"]

        print("\n" + "=" * 80)
        print(f"Test {i}: {name}")
        print(f"QUERY: {query}")
        session_id = f"session_{session_base}_{i}"

        result = await run_agent_query(talk_agent, query, user_id, session_id)
        tool_used = (result.get("tool_used") or "unknown").lower()
        tool_match = (expected_tool.lower() == tool_used)

        tool_output = result.get("tool_output")
        text = result.get("text", "")

        # If tool_output is None use text
        raw_response = tool_output if tool_output is not None else text

        # If target provided, compare
        if target_title:
            cmp = compare_target_to_response(target_title, raw_response)
            exact = cmp["exact_found"]
            best_ratio = cmp["best_ratio"]
            best_title = cmp["best_candidate_title"] or ""
        else:
            exact = None
            best_ratio = None
            best_title = None

        # use gemini to score similarity (target vs best_title or raw_response)
        if target_title:
            candidate_for_score = best_title or (str(raw_response)[:800])
            sim_score = gemini_similarity_score_sync(target_title, candidate_for_score, api_key=key)
        else:
            # for general searches we'll ask Gemini to score how relevant these results are to "sport and health"
            # We'll supply the query and the top text and let Gemini return 1..10
            candidate_for_score = str(raw_response)[:800] if raw_response else text[:800]
            sim_score = gemini_similarity_score_sync(query, candidate_for_score, api_key=key)

        # Print results compactly
        print(f"Tool USED by agent: {tool_used}")
        print(f"Expected tool: {expected_tool}  — Match: {tool_match}")
        if target_title:
            print(f"Target article: {target_title}")
            print(f"Exact found: {exact}")
            print(
                f"Best fuzzy ratio: {best_ratio:.3f} (0..1) — scaled 0..10 => {best_ratio * 10:.1f}" if best_ratio is not None else "No ratio")
            print("Best matched title/snippet:", json.dumps(cmp.get("best_candidate_title"), ensure_ascii=False))
        else:
            print("No explicit target for this query; showing top returned text snippet:")
            snippet = str(raw_response)[:800] if raw_response else text[:800]
            print(snippet)

        print(f"Gemini similarity score (1..10): {sim_score:.2f}")

    print("\nAll tests finished.\n")


# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    asyncio.run(evaluate_all())
