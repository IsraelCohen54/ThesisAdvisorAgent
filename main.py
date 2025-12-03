# main.py
import ast
import uuid
import json
import logging
import asyncio
from typing import Any
from google.adk.apps.app import App
from google.adk.runners import Runner
from app.config.settings import logger, THESIS_MINIMUM_LENGHT
from google.genai import types as gen_types
from app.core.agents import get_dialog_agent1
from google.adk.sessions import InMemorySessionService
from app.core.anylize_and_recommend import execute_debate_process

# --- Logging Setup ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)

# -----------------------
# Helper: The "Pretty Printer"
# -----------------------
def format_results_for_display(response_obj: Any) -> str:
    """
    Parses raw tool output (JSON, Dict string, or List) and returns a beautiful
    text format for the user.
    """
    if response_obj is None:
        return "No results found."

    # 1. PARSING: Try to turn strings back into Lists/Dicts
    if isinstance(response_obj, str):
        cleaned = response_obj.strip()
        try:
            # Case A: Proper JSON (Double quotes)
            if cleaned.startswith("{") or cleaned.startswith("["):
                response_obj = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                # Case B: Python Dict String (Single quotes - from tools.py)
                response_obj = ast.literal_eval(cleaned)
            except (ValueError, SyntaxError):
                pass  # It's just normal text, kept as string

    # 2. EXTRACTION: Get the list from inside the dict
    if isinstance(response_obj, dict):
        if "error" in response_obj:
            return f"❌ Error: {response_obj['error']}"
        # Grab content from 'result', 'organic_results', or return as is
        response_obj = response_obj.get("result", response_obj.get("organic_results", response_obj))

    # 3. FORMATTING: Iterate through the list and style it
    if isinstance(response_obj, list):
        if not response_obj:
            return "No relevant results found."

        lines = []
        for i, item in enumerate(response_obj, 1):
            if isinstance(item, dict):
                # Safely get fields
                title = item.get("title", "No Title")
                authors = item.get("authors") or item.get("AU")
                source = item.get("source") or item.get("journal")
                link = item.get("link") or item.get("url") or "No link"
                snippet = item.get("snippet", item.get("abstract", ""))

                # Cleanup snippet
                if snippet:
                    snippet = snippet[:250].replace("\n", " ") + "..."

                # Build the visual block
                lines.append(f"{i}. {title}")  # Plain text title

                # Add metadata with indentation
                if authors:
                    lines.append(f"   👤 Authors: {authors}")
                if source:
                    lines.append(f"   📰 Source:  {source}")
                if snippet:
                    lines.append(f"   📖 Snippet: {snippet}")
                lines.append(f"   🔗 Link:    {link}")
                lines.append("")  # Empty line between results
            else:
                # Fallback if item isn't a dict
                lines.append(f"{i}. {str(item)}")

        return "\n".join(lines)

    # 4. FALLBACK: Just return the string
    return str(response_obj)


# -----------------------
# Helper: parse tool output to structured object (list/dict) when possible
# -----------------------
def parse_tool_output_to_struct(obj):
    """
    Try to return a structured Python object (list/dict) from the tool output.
    If parsing fails, return the original object (string) so the pipeline still works.
    """
    if obj is None:
        return None

    if isinstance(obj, (list, dict)):
        return obj

    s = str(obj).strip()
    # Try JSON first (preferred)
    if s.startswith("{") or s.startswith("["):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

    # Try Python literal (single-quotes / other)
    try:
        return ast.literal_eval(s)
    except Exception:
        # not parseable -> return original string
        return s


# --- Main Async Logic ---
async def main():
    print("🎓 Thesis Advisor — Interactive Agent")

    # 1. Initialize System
    dialog_agent1 = get_dialog_agent1()
    app = App(name="agents", root_agent=dialog_agent1)
    runner = Runner(app=app, session_service=InMemorySessionService())

    user_id = "user_1"
    session_id = f"session_{uuid.uuid4().hex[:6]}"

    # Create session
    await runner.session_service.create_session(app_name="agents", user_id=user_id, session_id=session_id)

    thesis_input = input("\n📝 Enter your thesis/idea: ").strip()
    while thesis_input.__len__() < THESIS_MINIMUM_LENGHT: # defending against a miss click \ enter
        thesis_input = input("\n📝 Enter your thesis/idea (at least 10 characters): ").strip()

    while True:
        if not thesis_input:
            break

        print(f"\n🔎 Searching for: {thesis_input}...")

        content = gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=thesis_input)])

        tool_output = None
        full_text = []

        try:
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        # 1. Priority: Check for structured function response
                        if getattr(part, "function_response", None):
                            # Try to get 'result' directly if the tool returns a dict
                            resp = part.function_response.response
                            tool_output = resp.get("result") if isinstance(resp, dict) else resp

                        # 2. Capture text (in case agent talks)
                        if getattr(part, "text", None):
                            full_text.append(part.text)
        except Exception as e:
            print(f"❌ Error during search: {e}")
            break

        print("-" * 60)
        # Display Logic: Prefer Tool Output -> Then Agent Text
        if tool_output:
            print(format_results_for_display(tool_output))
        elif full_text:
            # Fallback: Sometimes the agent output IS the tool output string
            combined_text = "\n".join(full_text)
            print(format_results_for_display(combined_text))
        else:
            print("No results found.")
        print("-" * 60)

        # Human Loop
        choice = input("\n[Q]uit (exact match) / [C]ontinue (debate it) / [R]efine/replace thesis idea: ").strip().lower()

        if choice == 'q':
            print("If you would have another thesis idea, you are invited to try again!")
            break
        elif choice == 'c':
            if tool_output:
                references_obj = parse_tool_output_to_struct(tool_output)
            else:
                # if no structured tool output, prefer the full agent text (string)
                references_obj = "\n".join(full_text) if full_text else None


            # The debate process requires runner/user_id/session_id for the
            # Criteria Selection Dialog phase (which is asynchronous).
            await execute_debate_process(
                thesis_text=thesis_input,
                references_json=references_obj,
                runner=runner,
                user_id=user_id,
                session_id=session_id,
                blocking_mode="to_thread"
            )
            break
        elif choice == 'r':
            thesis_input = input("Enter refined thesis idea: ").strip()
            while thesis_input.__len__() < 5:  # defending against a miss click \ enter
                thesis_input = input("\n📝 Enter your thesis/idea: ").strip()
            continue
        else:
            thesis_input = input("Assuming you wanted to try again. \nPlease enter refined query: ").strip()
            continue


if __name__ == "__main__":
    asyncio.run(main())