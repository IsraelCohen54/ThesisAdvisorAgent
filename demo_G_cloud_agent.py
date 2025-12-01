# thesis_advisor_client.py
import vertexai
import uuid
import json
import asyncio
import logging
import ast
from typing import Any
import os

import sys



# Windows systems often default to this for console I/O
# We check if stdin is a console to apply the fix safely.
if sys.platform == "win32" and sys.stdin.isatty():
    # Force stdin and stdout to use utf-8 encoding
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')


os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "flowing-precept-479317-j4")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
# optional: ensure ADK uses Vertex backend (some examples set this env var)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
vertexai.init(project=os.environ["GOOGLE_CLOUD_PROJECT"], location=os.environ["GOOGLE_CLOUD_LOCATION"])

# Import ADK Components for local runner/session
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as gen_types

# Import the local components needed for the debate flow
from app.core.agents import get_dialog_agent1
from app.core.anylize_and_recommend import execute_debate_process

# --- Logging Setup ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ThesisAdvisor")
logger.setLevel(logging.INFO)

PROJECT_ID = "flowing-precept-479317-j4"
REGION = "us-central1"

# --- Helper: The "Pretty Printer" ---
def format_results_for_display(response_obj: Any) -> str:
    """
    Parses raw tool output (JSON, Dict string, or List) and returns a beautiful
    text format for the user. (Same as your original helper)
    """
    if response_obj is None:
        return "No results found."

    if isinstance(response_obj, str):
        cleaned = response_obj.strip()
        try:
            if cleaned.startswith("{") or cleaned.startswith("["):
                response_obj = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                response_obj = ast.literal_eval(cleaned)
            except (ValueError, SyntaxError):
                pass

    if isinstance(response_obj, dict):
        if "error" in response_obj:
            return f"❌ Error: {response_obj['error']}"
        response_obj = response_obj.get("result", response_obj.get("organic_results", response_obj))

    if isinstance(response_obj, list):
        if not response_obj:
            return "No relevant results found."

        lines = ["\n📚 Found Search Results:"]
        for i, item in enumerate(response_obj, 1):
            if isinstance(item, dict):
                title = item.get("title", "No Title")
                authors = item.get("authors") or item.get("AU")
                link = item.get("link") or item.get("url") or "No link"
                snippet = item.get("snippet", item.get("abstract", ""))[:250].replace("\n", " ") + "..."

                lines.append(f"{i}. {title}")
                if authors: lines.append(f"   👤 Authors: {authors}")
                lines.append(f"   📖 Snippet: {snippet}")
                lines.append(f"   🔗 Link:    {link}\n")
            else:
                lines.append(f"{i}. {str(item)}")
        return "\n".join(lines)

    return str(response_obj)


# --- Main Async Logic ---
async def main():
    print("🎓 Thesis Advisor — Interactive Agent System")
    vertexai.init(project=PROJECT_ID, location=REGION)
    # 1. Initialize Local System (App and Session for local debate)
    dialog_agent1 = get_dialog_agent1()
    app = App(name="agents", root_agent=dialog_agent1)
    runner = Runner(app=app, session_service=InMemorySessionService())

    user_id = "user_1"
    session_id = f"session_{uuid.uuid4().hex[:6]}"
    await runner.session_service.create_session(app_name="agents", user_id=user_id, session_id=session_id)

    thesis_input = input("\n📝 Enter your thesis/idea: ").strip()

    while True:
        if not thesis_input:
            break

        print(f"\n🔎 Searching for: {thesis_input}...")

        content = gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=thesis_input)])

        # Variables to capture the two types of successful output
        raw_tool_output = None
        final_agent_text = []

        try:
            # Call the DEPLOYED Agent using the local Runner
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        # 1. Capture RAW TOOL DATA (for debate bridge)
                        if part.function_response:
                            resp = part.function_response.response
                            # Capture the RAW list of dicts from the tool output
                            raw_tool_output = resp.get("result") if isinstance(resp, dict) else resp
                            print("\n[Function call trace completed]")  # Trace debug

                        # 2. Capture FINAL FORMATTED TEXT (for user display)
                        if part.text:
                            # Print the final text as it is streamed from the deployed agent
                            print(part.text, end="", flush=True)
                            final_agent_text.append(part.text)  # Keep full text for reference

        except Exception as e:
            print(f"❌ Error during search: {e}")
            break

        print("\n" + "-" * 60)

        # --- PHASE 1 DISPLAY & BRIDGE LOGIC ---

        # If the deployed agent returned only raw tool output (the old way), display it cleanly.
        if raw_tool_output and not final_agent_text:
            print(format_results_for_display(raw_tool_output))
        elif not raw_tool_output and not final_agent_text:
            print("No results found.")

        # Prepare the reference bridge for the debate agents
        # PRIORITY 1: The raw list of dicts (best context for debate)
        # PRIORITY 2: The final text (if no raw data was captured)
        references_for_debate = str(raw_tool_output) if raw_tool_output else "".join(final_agent_text)

        # --- PHASE 2: HUMAN LOOP ---
        choice = input("\n[Q]uit / [C]ontinue (debate it) / [R]efine thesis idea: ").strip().lower()

        if choice == 'q':
            print("If you would have another thesis idea, you are invited to try again!")
            break
        elif choice == 'c':
            print("\n\n--- 🗣️ LAUNCHING DEBATE FLOW ---")
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