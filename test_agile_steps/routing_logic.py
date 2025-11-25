# In main.py
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import logging
import os
from app.core.agents import ThesisAdvocatorAgent


def cleanup_logs():
    for log_file in ["logger.log", "web.log", "tunnel.log"]:
        if os.path.exists(log_file):
            os.remove(log_file)


cleanup_logs()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ThesisAdvocator")


def initialize_runner():
    # Check for simplified key
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("FATAL: GEMINI_API_KEY not found.")
        return None

    # 1. Create Session
    session_service = InMemorySessionService()

    # 2. Create Agent (Clean init)
    advocator_agent = ThesisAdvocatorAgent()

    # 3. Create App
    advocator_app = App(
        name="ThesisAdvocatorApp",
        root_agent=advocator_agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )

    # 4. Create Runner
    runner = Runner(
        app=advocator_app,
        session_service=session_service,
    )
    return runner


def inspect_response(response, step_name):
    """
    Robustly inspects an ADK Agent response object to find text and tool calls.
    Works across different ADK versions by checking attributes dynamically.
    """
    print(f"\n--- 🔍 INSPECTING: {step_name} ---")

    # 1. Try to print the TEXT response
    text_found = False
    # Check common attributes for text
    for attr in ['text', 'output', 'content']:
        val = getattr(response, attr, None)
        if val:
            print(f"🤖 Agent Answer: {val}")
            text_found = True
            break

    if not text_found:
        print("🤖 Agent Answer: [No text output found]")

    # 2. Try to find TOOL CALLS
    # Tool calls might be on the response object, or nested in 'steps' or 'candidates'
    tool_found = False

    # Check direct attribute
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"🛠️ Tool Called (Direct): {response.tool_calls[0].function.name}")
        tool_found = True

    # Check nested steps/history (common in ADK runners)
    if not tool_found and hasattr(response, 'steps'):
        for step in response.steps:
            if hasattr(step, 'tool_calls') and step.tool_calls:
                print(f"🛠️ Tool Called (In Step): {step.tool_calls[0].function.name}")
                tool_found = True
                break

    if not tool_found:
        print("🛠️ Tool Status: No external tool was triggered.")


def run_tests(runner: Runner):
    # Test 1: Bio
    bio_thesis = "Please find recent articles discussing the ethical implications of human organoids."

    # Use standard .run()
    response_1 = runner.run(user_id="1", session_id="session_bio_1", new_message=bio_thesis)
    inspect_response(response_1, "Biomedical Thesis")

    # Test 2: General
    general_thesis = "What are the major academic disagreements on the causes of the Renaissance?"

    response_2 = runner.run(user_id="1", session_id="session_gen_1", new_message=general_thesis)
    inspect_response(response_2, "General Thesis")


if __name__ == "__main__":
    runner = initialize_runner()
    if runner:
        run_tests(runner)
