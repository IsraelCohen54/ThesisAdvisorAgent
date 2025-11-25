# In main.py
import logging
import os
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from app.core.agents import ThesisAdvocatorAgent
from google.genai import types


def cleanup_logs():
    """Cleans up log files from previous runs."""
    for log_file in ["logger.log", "web.log", "tunnel.log"]:
        if os.path.exists(log_file):
            os.remove(log_file)
            print(f"🧹 Cleaned up {log_file}")


# Configure logging with DEBUG log level (better for production tracing)
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,  # Use DEBUG for detailed tracing
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)
# Setup logging
logger = logging.getLogger("ThesisAdvocator")


def initialize_runner():
    """Initializes the stateful App and Runner."""

    # Check for API Key
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("FATAL: GEMINI_API_KEY not found. Please set it in PyCharm configurations.")
        return None

    # 1. Create the Session Service (The persistence layer)
    session_service = InMemorySessionService()
    logger.info("✅ Session Service (Memory) created.")

    # 2. Create the Agent (The root_agent for the App)
    # We remove the session_service argument from the ThesisAdvocatorAgent init
    # because the session is managed by the App/Runner, not the Agent itself.
    advocator_agent = ThesisAdvocatorAgent(session_service)
    logger.info("✅ Thesis Advocator Agent (Root Agent) created.")

    # 3. Wrap in Resumable App (THE KEY FOR LONG-RUNNING OPERATIONS)
    advocator_app = App(
        name="ThesisAdvocatorApp",
        root_agent=advocator_agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    logger.info("✅ Resumable App created.")

    # 4. Create Runner (Manages the conversation flow)
    advocator_runner = Runner(
        app=advocator_app,  # Pass the App!
        session_service=session_service,
    )
    logger.info("✅ Runner initialized.")

    return advocator_runner, session_service


def run_tests(runner: Runner):
    """Executes the validation tests using the Runner."""

    # Use a unique session ID for the user's debate
    USER_SESSION_ID = "debate_session_123"

    # 1. Test 1: Biomedical Thesis (Should call PubMed MCP)
    print("\n--- TEST 1: Biomedical Thesis (MCP) ---")
    bio_thesis = "Please find recent articles discussing the ethical implications of human organoids."
    response_1 = runner.run(
        session_id=USER_SESSION_ID,
        prompt=bio_thesis
    )
    print(f"Router Response: {response_1.text}")
    print(f"Tool Status: {response_1.status.name}")  # Status should be 'DONE' or 'PAUSED' if waiting for approval

    # 2. Test 2: General Thesis (Should call Google Scholar)
    print("\n--- TEST 2: General Thesis (Scholar) ---")
    general_thesis = "What are the major academic disagreements on the causes of the Renaissance?"
    response_2 = runner.run(
        session_id="debate_session_2",  # Use a different session for parallel tracking
        prompt=general_thesis
    )
    print(f"Router Response: {response_2.text}")
    print(f"Tool Status: {response_2.status.name}")


if __name__ == "__main__":
    runner_data = initialize_runner()
    if runner_data:
        runner, session_service = runner_data
        run_tests(runner)
