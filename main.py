import os
import uuid
import json
import asyncio
import logging
import ast  # <--- NEW: Needed to parse the python dictionary string safely
from typing import Any

from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.google_llm import Gemini
from google.genai import types as gen_types

from app.core.agents import get_dialog_agent1

from app.core.anylize_and_recommend import execute_debate_process

# --- Logging Setup ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ThesisAdvisor")
logger.setLevel(logging.INFO)


# --- Helper: The "Pretty Printer" ---
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
                # Case B: Python Dict String (Single quotes - typical from your tools.py)
                response_obj = ast.literal_eval(cleaned)
            except (ValueError, SyntaxError):
                pass  # It's just normal text, keep it as string

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


# --- Helper: Debater ---
def run_debaters(thesis_text: str, references_text: str):
    """Run the Debater analysis (Sync, because it's a simple API call)"""
    print("\n⚖️  Running Debaters...")
    model = Gemini(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))

    pro_prompt = f"You are a PRO debater. Analyze this thesis based on the references.\nThesis: {thesis_text}\nreferences: {references_text}\n\nProvide 3 strong PRO points and how can you manage risk factors well."
    con_prompt = f"You are a CON debater. Analyze this thesis based on the references.\nThesis: {thesis_text}\nreferences: {references_text}\n\nProvide 3 strong CON points and risk factors."

    try:
        pro_resp = model.api_client.models.generate_content(model="gemini-2.5-flash", contents=pro_prompt)
        con_resp = model.api_client.models.generate_content(model="gemini-2.5-flash", contents=con_prompt)

        print(f"\n✅ --- PRO Argument ---\n{pro_resp.text}")
        print(f"\n❌ --- CON Argument ---\n{con_resp.text}")

        synth_prompt = f"Synthesize these two arguments into a final recommendation:\nPRO: {pro_resp.text}\nCON: {con_resp.text}"
        synth_resp = model.api_client.models.generate_content(model="gemini-2.5-flash", contents=synth_prompt)
        print(f"\n📝 --- FINAL SYNTHESIS ---\n{synth_resp.text}")

    except Exception as e:
        print(f"Debater error: {e}")


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
                        if part.function_response:
                            # Try to get 'result' directly if the tool returns a dict
                            resp = part.function_response.response
                            tool_output = resp.get("result") if isinstance(resp, dict) else resp

                        # 2. Capture text (in case agent talks)
                        if part.text:
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
            references = str(tool_output) if tool_output else "\n".join(full_text)
            final_result = await execute_debate_process(
                thesis_text=thesis_input,
                references_json=references,
                runner=runner,
                user_id=user_id,
                session_id=session_id,
                api_key=os.getenv("GEMINI_API_KEY"),
                blocking_mode="to_thread"
            )
            print(final_result)
            break
        elif choice == 'r':
            thesis_input = input("Enter refined thesis idea: ").strip()
            continue
        else:
            thesis_input = input("Assuming you wanted to try again. \nPlease enter refined query: ").strip()
            continue


if __name__ == "__main__":
    asyncio.run(main())
