import os
import uuid
import json
import asyncio
import logging
from typing import Any, List

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.google_llm import Gemini
from google.genai import types as gen_types

from app.core.agents import get_talk_agent
from app.infrastructure.tools import PubMedTool, GoogleScholarTool

# --- Logging Setup ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ThesisAdvocator")
logger.setLevel(logging.INFO)


# --- Helper: Output Formatting ---
def format_results_for_display(response_obj: Any) -> str:
    """Cleanly formats the messy output that might be dict, list, string, or JSON-string."""
    if response_obj is None:
        return "No results found."

    # 1. Unwrap JSON strings
    if isinstance(response_obj, str):
        try:
            # If the string is actually a JSON repr of a dict/list, parse it
            if response_obj.strip().startswith("{") or response_obj.strip().startswith("["):
                response_obj = json.loads(response_obj)
            else:
                return response_obj  # It's just a normal string text
        except:
            pass  # Keep as string

    # 2. Extract 'result' key if it exists
    if isinstance(response_obj, dict):
        if "error" in response_obj:
            return f"Error: {response_obj['error']}"
        response_obj = response_obj.get("result", response_obj)

    # 3. Handle List of Dicts (The ideal format)
    if isinstance(response_obj, list):
        lines = []
        for i, item in enumerate(response_obj, 1):
            if isinstance(item, dict):
                title = item.get("title", "No Title")
                source = item.get("source") or item.get("journal", "")
                link = item.get("link") or item.get("url", "")
                snippet = item.get("snippet", "")[:200].replace("\n", " ")

                lines.append(f"{i}. **{title}**")
                if source: lines.append(f"   Source: {source}")
                if link: lines.append(f"   Link: {link}")
                if snippet: lines.append(f"   Snippet: {snippet}...")
                lines.append("")
            else:
                lines.append(f"{i}. {str(item)}")
        return "\n".join(lines)

    # 4. Fallback
    return str(response_obj)


# --- Helper: Debater ---
def run_debaters(thesis_text: str, evidence_text: str):
    """Run the Debater analysis (Sync, because it's a simple API call)"""
    print("\n⚖️  Running Debaters...")
    model = Gemini(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))

    # We ask for plain text to avoid parsing issues
    pro_prompt = f"You are a PRO debater. Analyze this thesis based on the evidence.\nThesis: {thesis_text}\nEvidence: {evidence_text}\n\nProvide 3 strong PRO points."
    con_prompt = f"You are a CON debater. Analyze this thesis based on the evidence.\nThesis: {thesis_text}\nEvidence: {evidence_text}\n\nProvide 3 strong CON points and risk factors."

    try:
        # Use direct API client for speed
        pro_resp = model.api_client.models.generate_content(model="gemini-2.5-flash", contents=pro_prompt)
        con_resp = model.api_client.models.generate_content(model="gemini-2.5-flash", contents=con_prompt)

        print(f"\n✅ --- PRO Argument ---\n{pro_resp.text}")
        print(f"\n❌ --- CON Argument ---\n{con_resp.text}")

        # Synthesis
        synth_prompt = f"Synthesize these two arguments into a final recommendation:\nPRO: {pro_resp.text}\nCON: {con_resp.text}"
        synth_resp = model.api_client.models.generate_content(model="gemini-2.5-flash", contents=synth_prompt)
        print(f"\n📝 --- FINAL SYNTHESIS ---\n{synth_resp.text}")

    except Exception as e:
        print(f"Debater error: {e}")


# --- Main Async Logic ---
async def main():
    print("🎓 Thesis Advocator — Interactive Agent")

    # 1. Initialize System (ONCE)
    talk_agent = get_talk_agent()
    app = App(name="ThesisApp", root_agent=talk_agent)
    runner = Runner(app=app, session_service=InMemorySessionService())

    user_id = "user_1"
    session_id = f"session_{uuid.uuid4().hex[:6]}"

    # Create session explicitly
    await runner.session_service.create_session(app_name="ThesisApp", user_id=user_id, session_id=session_id)

    thesis_input = input("\n📝 Enter your thesis/idea: ").strip()

    while True:
        if not thesis_input:
            break

        print(f"\n🔎 Searching for: {thesis_input}...")

        # 2. Run the Agent
        # We use run_async to stay in the loop
        content = gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=thesis_input)])

        tool_output = None
        full_text = []

        try:
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
                # Capture function responses (The structured data)
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.function_response:
                            # This is the raw data from the tool
                            tool_output = part.function_response.response.get("result")
                        if part.text:
                            full_text.append(part.text)
        except Exception as e:
            print(f"❌ Error during search: {e}")
            break

        # 3. Display Results
        # If we got raw tool data, format it. If not, show whatever text the agent generated.
        if tool_output:
            final_display = format_results_for_display(tool_output)
        elif full_text:
            final_display = "\n".join(full_text)
        else:
            final_display = "No results found (Agent did not return text or data)."

        print("-" * 60)
        print(final_display)
        print("-" * 60)

        # 4. Human Loop
        choice = input("\n[Y]es (exact match) / [N]o (debate it) / [R]efine / [Q]uit: ").strip().lower()

        if choice == 'y':
            follow = input("Found it! [S]uggest new idea / [Q]uit: ").lower()
            if follow == 's':
                thesis_input = input("Enter new idea: ").strip()
                continue
            break
        elif choice == 'n':
            # Run debate on the current evidence
            run_debaters(thesis_input, final_display)
            break
        elif choice == 'r':
            thesis_input = input("Enter refined query: ").strip()
            continue  # Loop again with new input
        else:
            print("Goodbye.")
            break


if __name__ == "__main__":
    # Run the main async loop
    asyncio.run(main())