# demo_G_cloud_agent.py
import os
import uuid
import asyncio
import logging
import vertexai
from typing import Any, List, Optional, Dict

# ADK Components
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.genai import types as gen_types
from google.adk.sessions import InMemorySessionService

# local components
from app.core.agents import get_dialog_agent1
from app.core.anylize_and_recommend import execute_debate_process
from app.config.settings import logger, PROJECT_ID, REGION, THESIS_MINIMUM_LENGHT, USER_ID, SESSION_ID
from app.function_helpers.cloud_helpers import normalize_tool_output, pretty_display, run_tool_and_get_result

# --- Environment Setup (Must be before ADK imports) ---
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", REGION)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
vertexai.init(project=os.environ["GOOGLE_CLOUD_PROJECT"], location=os.environ["GOOGLE_CLOUD_LOCATION"])

# --- Logging Setup ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')
logger.setLevel(logging.INFO)




# --- Main Async Logic ---
async def main():
    print("🎓 Thesis Advisor — Interactive Agent System")

    # 1. Initialize Local System (App and Session for local debate)
    dialog_agent1 = get_dialog_agent1()
    app = App(name="agents", root_agent=dialog_agent1)
    runner = Runner(app=app, session_service=InMemorySessionService())

    await runner.session_service.create_session(app_name="agents", user_id=USER_ID, session_id=SESSION_ID)

    thesis_input = input("\n📝 Enter your thesis/idea: ").strip()
    while len(thesis_input) < THESIS_MINIMUM_LENGHT:  # defending against a miss click \ enter
        thesis_input = input("\n📝 Enter your thesis/idea (at least 10 characters): ").strip()

    while True:
        print(f"\n🔎 Searching for: {thesis_input}...")

        content = gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=thesis_input)])

        # Variables to capture the two types of successful output
        raw_tool_output: Any = None
        final_agent_text: List[str] = []

        try:
            # Call the DEPLOYED Agent using the local Runner
            async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        # 0. Detect Tool Request (function_call) -> Execute Tool Locally
                        fc = getattr(part, "function_call", None)
                        if fc:
                            name = getattr(fc, "name", None)

                            # We also check for 'args' as some SDK versions use different names
                            args = getattr(fc, "arguments", None) or getattr(fc, "args", {})

                            # We perform one final check to ensure it's a dictionary before passing it
                            if not isinstance(args, dict):
                                args = {}

                            print(f"Model requested tool: {name} args: {args}")

                            # 1. Execute the tool locally (you need to define or import this function)
                            tool_result = run_tool_and_get_result(name, args)

                            # 2. IMMEDIATELY NORMALIZE the result to clean list/dict structure
                            raw_tool_output = normalize_tool_output(tool_result)

                            print("\n[Function call trace completed]")  # Trace debug, this is what you are seeing
                            continue

                        if getattr(part, "text", None):
                            text_content = part.text

                            # --- CRITICAL NEW LOGIC: Intercept raw structured text ---
                            maybe_parsed = normalize_tool_output(text_content)

                            # If parsing the text results in a list or dict, it means the agent
                            # is streaming the raw tool result here. Capture it and silence the print.
                            if isinstance(maybe_parsed, (list, dict)):
                                # Capture the structured data for pretty_display later
                                raw_tool_output = maybe_parsed
                                # Do NOT print it. Skip the rest of the loop for this part.
                                continue
                                # --- END CRITICAL NEW LOGIC ---

                            # If it's not structured, it's genuine narrative text. Stream it.
                            print(text_content, end="", flush=True)
                            final_agent_text.append(text_content)


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
                user_id=USER_ID,
                session_id=SESSION_ID,
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
