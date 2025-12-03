# cloud_helpers
import re
import ast
import json
from typing import Any, Optional, List, Dict
from app.infrastructure.tools import GoogleScholarTool, PubMedTool
from app.config.settings import logger

# Instantiate your tools once if possible, or inside the function if needed
# Instantiating here is usually cleaner for simple clients:
google_scholar_tool = GoogleScholarTool()
pubmed_tool = PubMedTool()
# -----------------------
# Parsing & Pretty helpers
# -----------------------
def safe_parse_string(s: Any) -> Any:
    """
    Try to turn `s` (string/bytes) into Python object:
    1) json.loads
    2) ast.literal_eval
    3) find first {...} or [...] substring and attempt parsing again
    If all fail, return cleaned string.
    """
    if s is None:
        return None

    # already structured
    if isinstance(s, (dict, list)):
        return s

    # normalize bytes
    if isinstance(s, (bytes, bytearray)):
        try:
            s = s.decode("utf-8")
        except Exception:
            s = str(s)

    if not isinstance(s, str):
        return s

    s_strip = s.strip()
    if not s_strip:
        return s_strip

    # try JSON first
    try:
        return json.loads(s_strip)
    except Exception:
        pass

    # try python-literal (single quotes)
    try:
        return ast.literal_eval(s_strip)
    except Exception:
        pass

    # try to locate first balanced {...} or [...] block and parse that
    # This helps if I get outer quoting like: "{'result': [...]}"

    # regex to find first { ... } or [ ... ] pair (greedy-ish but ok for our outputs)
    match = re.search(r'(\{.*}|\[.*])', s_strip, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        # try json then literal eval
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return ast.literal_eval(candidate)
            except Exception:
                pass

    # nothing parsed — return original trimmed string
    return s_strip


# Func for Parsing nested string:
def normalize_tool_output(obj: Any) -> Any:
    """
    Repeatedly unwrap nested wrappers until we get a native list/dict or stable primitive.
    Handles shapes like:
      {'pubmed_execute_response': {'result': "{'result': [...] }"}}
      {'result': "{'result': [...]}"}
      '{"result": "[{...}]"}'
    """
    current = obj
    MAX_ITER = 6 # unwrap depth
    for _ in range(MAX_ITER):
        # unwrap single-key "*_response" wrappers
        if isinstance(current, dict) and len(current) == 1:
            k = next(iter(current))
            if isinstance(k, str) and k.endswith("_response"):
                current = current[k]
                continue

        # if dict with 'result' or 'organic_results' -> take it
        if isinstance(current, dict) and "result" in current:
            current = current["result"]
            continue
        if isinstance(current, dict) and "organic_results" in current:
            current = current["organic_results"]
            continue

        # if a string-like object -> try parsing it
        if isinstance(current, (str, bytes, bytearray)):
            parsed = safe_parse_string(current)
            # if parsed turned into something new, continue loop
            if parsed is not None and parsed != current:
                current = parsed
                continue
            # couldn't parse further
            break

        # if list -> normalize elements (but don't try to flatten)
        if isinstance(current, list):
            # ensure each element is normalized once (parse string elements if needed)
            new_list = []
            for el in current:
                if isinstance(el, (str, bytes, bytearray)):
                    maybe = safe_parse_string(el)
                    new_list.append(maybe if maybe is not None else el)
                else:
                    new_list.append(el)
            return new_list

        # if dict (but not parsed 'result' case), attempt to normalize its values
        if isinstance(current, dict):
            # normalize values but return dict
            return {k: normalize_tool_output(v) for k, v in current.items()}

        # else primitive -> return
        break

    return current


def pretty_display(obj: Any, max_snippet: Optional[int] = None) -> str:
    """
    Return a human-friendly multiline string for obj (list/dict/str).
     - If list of dicts -> numbered Title/Authors/Source/Link/Snippet blocks.
     - If dict -> pretty JSON (indent=2)
     - else -> str(obj)
    """
    norm = normalize_tool_output(obj)

    # If list -> format as numbered list
    if isinstance(norm, list):
        if not norm:
            return "No relevant results found."

        lines = []
        for idx, item in enumerate(norm, start=1):
            # if item is string, try parse once more
            if isinstance(item, str):
                maybe = safe_parse_string(item)
                if isinstance(maybe, (list, dict)):
                    item = maybe

            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or "No Title"
                authors = item.get("authors") or item.get("author") or item.get("AU") or ""
                source = item.get("source") or item.get("journal") or item.get("publisher") or ""
                link = item.get("link") or item.get("url") or item.get("pmid") or ""
                snippet = item.get("snippet") or item.get("abstract") or item.get("summary") or ""

                clean_snip = " ".join(str(snippet).split())
                if max_snippet and len(clean_snip) > max_snippet:
                    clean_snip = clean_snip[:max_snippet].rstrip()

                lines.append(f"{idx}. {title}")
                if authors:
                    lines.append(f"   Authors: {authors}")
                if source:
                    lines.append(f"   Source:  {source}")
                if link:
                    lines.append(f"   Link:    {link}")
                if clean_snip:
                    lines.append(f"   Snippet: {clean_snip}")
                lines.append("")  # blank line between items
            else:
                lines.append(f"{idx}. {str(item)}")
                lines.append("")

        # remove trailing blank line
        if lines and lines[-1] == "":
            lines = lines[:-1]
        return "\n".join(lines)

    # If dict -> pretty JSON
    if isinstance(norm, dict):
        try:
            return json.dumps(norm, indent=2, ensure_ascii=False)
        except Exception:
            return str(norm)

    # fallback -> string
    return str(norm)


def run_tool_and_get_result(name: str, args: Dict[str, Any]) -> Any:
    """
    Executes the appropriate local tool based on the name and arguments
    provided by the Agent Model.
    """
    query = args.get('query') or args.get('text') or ''
    if not query:
        logger.warning(f"Tool {name} called without a valid query.")
        return {"error": f"Missing query for tool {name}"}


    # --- Tool Dispatcher ---
    if name == "google_scholar_execute":
        # Call the execute method of the GoogleScholarTool instance
        return google_scholar_tool.execute(query=query)

    elif name == "pubmed_execute":
        # The logic for pubmed_execute in agents.py already handles the fallback.
        # Since your main agent is calling the *function* name, you should
        # map to the logic that matches that function's behavior, which includes the fallback.
        # However, for simplicity and to stay consistent with the other tool,
        # let's assume the model is expected to handle the simple tool result first.

        # FIX: Call the execute method of the PubMedTool instance
        # NOTE: If you need the fallback, you should call the `pubmed_execute` function
        # imported from agents.py instead of the raw tool class. Let's keep it simple
        # and assume the execution of the raw tool is what's desired for a simple client.
        return pubmed_tool.execute(query=query)

    else:
        logger.error(f"Unknown tool requested by model: {name}")
        return {"error": f"Unknown tool: {name}"}