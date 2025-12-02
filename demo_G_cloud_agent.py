# thesis_advisor_client.py
import vertexai
import uuid
import json
import asyncio
import logging
import ast
from typing import Any, Dict, List, Optional
import os
import sys
import re

# Import ADK Components
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as gen_types

# Import local components
from app.core.agents import get_dialog_agent1
from app.core.anylize_and_recommend import execute_debate_process

from app.config.settings import logger, PROJECT_ID, REGION

# --- Environment Setup (Must be before ADK imports) ---
if sys.platform == "win32" and sys.stdin.isatty():
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", REGION)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
vertexai.init(project=os.environ["GOOGLE_CLOUD_PROJECT"], location=os.environ["GOOGLE_CLOUD_LOCATION"])

# --- Logging Setup ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)


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
    match = re.search(r'(\{.*\}|\[.*\])', s_strip, flags=re.DOTALL)
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


def normalize_tool_output(obj: Any) -> Any:
    """
    Repeatedly unwrap nested wrappers until we get a native list/dict or stable primitive.
    Handles shapes like:
      {'pubmed_execute_response': {'result': "{'result': [...] }"}}
      {'result': "{'result': [...]}"}
      '{"result": "[{...}]"}'
    """
    current = obj
    MAX_ITER = 6
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


# --- Main Async Logic ---
async def main():
    print("🎓 Thesis Advisor — Interactive Agent System")

    # 1. Initialize Local System (App and Session for local debate)
    dialog_agent1 = get_dialog_agent1()
    app = App(name="agents", root_agent=dialog_agent1)
    runner = Runner(app=app, session_service=InMemorySessionService())

    user_id = "user_1"
    session_id = f"session_{uuid.uuid4().hex[:6]}"
    await runner.session_service.create_session(app_name="agents", user_id=user_id, session_id=session_id)

    thesis_input = input("\n📝 Enter your thesis/idea: ").strip()
    while len(thesis_input) < 5:  # defending against a miss click \ enter
        thesis_input = input("\n📝 Enter your thesis/idea: ").strip()

    while True:
        print(f"\n🔎 Searching for: {thesis_input}...")

        content = gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=thesis_input)])

        # Variables to capture the two types of successful output
        raw_tool_output: Any = None
        final_agent_text: List[str] = []

        try:
            # Call the DEPLOYED Agent using the local Runner
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        # 1. Capture RAW TOOL DATA (for debate bridge)
                        if getattr(part, "function_response", None):
                            resp = part.function_response.response

                            # Normalize & unwrap nested string/dict wrappers robustly
                            parsed = normalize_tool_output(resp)
                            parsed = normalize_tool_output(parsed)  # second pass in case of nested 'result' string

                            raw_tool_output = parsed
                            print("\n[Function call trace completed]")  # Trace debug

                            # IMPORTANT: Do NOT print part.text when there was a function_response --
                            # that text is usually a stringified tool result (one-line) and we will
                            # pretty-print the parsed `raw_tool_output` after the stream finishes.
                            continue

                        # 2. Capture FINAL FORMATTED TEXT (for user display) ONLY when no function_response.
                        if getattr(part, "text", None):
                            # Print the final text as it is streamed from the deployed agent (only
                            # when it's genuine human/scaffold text, not a tool JSON dump).
                            print(part.text, end="", flush=True)
                            final_agent_text.append(part.text)  # Keep full text for reference


        except Exception as e:
            print(f"❌ Error during search: {e}")
            break

        print("\n" + "=" * 60)

        # --- PHASE 1 DISPLAY & BRIDGE LOGIC (MUST BE OUTSIDE THE ASYNC LOOP) ---
        # If the agent returned raw structured data (list/dict), prefer it and pretty display:
        if raw_tool_output and not final_agent_text:
            print(pretty_display(raw_tool_output, max_snippet=500))
        elif not raw_tool_output and not final_agent_text:
            print("No results found.")

        print("\n" + "=" * 60)

        # Prepare the reference bridge for the debate agents
        # PRIORITY 1: The raw structured object (best for debate agents)
        # PRIORITY 2: The final text (if no raw data was captured)
        references_for_debate = raw_tool_output if raw_tool_output else ("\n".join(final_agent_text) if final_agent_text else None)

        # --- PHASE 2: HUMAN LOOP ---
        choice = input("\n[Q]uit / [C]ontinue (debate it) / [R]efine thesis idea: ").strip().lower()

        if choice == 'q':
            print("If you would have another thesis idea, you are invited to try again!")
            break
        elif choice == 'c':
            print("\n\n--- 🗣️ LAUNCHING DEBATE FLOW ---")
            # pass structured object (list/dict) if available; otherwise pass text
            await execute_debate_process(
                thesis_text=thesis_input,
                references_json=references_for_debate,
                runner=runner,  # Pass the local runner for criteria dialogue
                user_id=user_id,
                session_id=session_id,
                blocking_mode="to_thread"  # Ensures the debate agents run without blocking
            )
            break  # Debate completed
        elif choice == 'r':
            thesis_input = input("Enter refined thesis idea: ").strip()
            continue
        else:
            thesis_input = input("Invalid choice. Please enter refined query: ").strip()
            continue


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
