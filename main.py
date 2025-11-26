import logging
import os
import asyncio
import uuid
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as gen_types

from app.core.agents import get_router_agent


# Logging Setup
def cleanup_logs():
    for log_file in ["logger.log", "web.log", "tunnel.log"]:
        if os.path.exists(log_file):
            os.remove(log_file)


cleanup_logs()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ThesisAdvocator")


# --- Helpers ---

def create_content_message(text: str) -> gen_types.Content:
    """Wraps text into ADK Content format."""
    return gen_types.Content(
        role="user",
        parts=[gen_types.Part.from_text(text=text)]
    )


async def _create_session_async(runner, user_id, session_id, app_name):
    """Async helper for session creation."""
    await runner.session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )


def initialize_system():
    """Sets up the App and Runner."""
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("FATAL: GEMINI_API_KEY not found.")
        return None, None

    agent = get_router_agent()

    app = App(
        name="ThesisAdvocatorApp",
        root_agent=agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )

    runner = Runner(
        app=app,
        session_service=InMemorySessionService(),
    )
    return runner, app.name


# --- Main Application Loop ---

def run_thesis_advocator():
    runner, app_name = initialize_system()
    if not runner:
        return

    # User & Session Setup
    USER_ID = "user_1"
    SESSION_ID = f"session_{uuid.uuid4().hex[:8]}"

    # 1. Create Session Sync
    asyncio.run(_create_session_async(runner, USER_ID, SESSION_ID, app_name))
    print(f"🎓 Thesis Advocator Initialized (Session: {SESSION_ID})")

    # 2. Get Initial Input
    thesis_input = input("\n📝 Enter your thesis/idea: ").strip()
    if not thesis_input:
        print("Empty input. Exiting.")
        return

    # 3. Research Loop (Refinement)
    approved = False
    current_query = thesis_input

    while not approved:
        print(f"\n🔎 Routing and Searching for: '{current_query}'...")

        # Run Agent
        message = create_content_message(current_query)
        events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=message)

        # Capture Output
        full_text_output = []
        tool_used = "None"

        for event in events:
            # Check for text
            if hasattr(event, "content") and event.content:
                for part in event.content.parts or []:
                    if part.text:
                        full_text_output.append(part.text)

            # Check for tool usage
            if hasattr(event, "tool_calls") and event.tool_calls:
                tool_used = event.tool_calls[0].function.name

        # Display Results
        print(f"\n🛠️ Tool Used: {tool_used}")
        print("-" * 40)
        final_response = "\n".join(full_text_output)
        print(final_response)
        print("-" * 40)

        # 4. Human Approval
        choice = input("\n🤔 Are these results relevant? (y = yes / n = no, refine / q = quit): ").lower().strip()

        if choice in ['y', 'yes']:
            approved = True
            print("\n✅ Evidence Approved.")
        elif choice in ['q', 'quit']:
            print("Exiting.")
            return
        else:
            current_query = input("🔄 Enter refined keywords or query: ").strip()
            print("Re-running search...")

    # 5. (Future) Debate Phase
    print("\n🚀 Starting Multi-Agent Debate Phase...")
    print(f"Context: {thesis_input}")
    print("Delegating to Pro/Con Agents... (Not implemented in this demo)")
    # Here you would call the Debater Agents using the data collected above


if __name__ == "__main__":
    run_thesis_advocator()