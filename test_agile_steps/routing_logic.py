# In main.py
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import logging
import os
from app.core.agents import ThesisAdvocatorAgent
from google.genai import types



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
    print(f"\n--- 🔍 INSPECTING: {step_name} ---")

    # If response is a generator/iterable, iterate and keep the last event
    last_event = None
    try:
        for event in response:
            last_event = event
    except Exception as e:
        print(f"🛑 Execution Error during iteration: {e}")
        return

    if last_event is None:
        print("⚠️ Runner returned no events.")
        return

    # Many ADK events have .messages (list), each message has .content, .tool_calls, etc.
    # 1) messages
    if hasattr(last_event, "messages") and getattr(last_event, "messages"):
        last_msg = last_event.messages[-1]
        # Content may be a Content-like object
        if hasattr(last_msg, "content") and last_msg.content:
            # content.parts is an array of Part objects
            parts = getattr(last_msg.content, "parts", None)
            if parts:
                # join all text parts
                text_parts = []
                for p in parts:
                    try:
                        text_parts.append(p.text)
                    except Exception:
                        # fallback: str(part)
                        text_parts.append(str(p))
                print("🤖 Last Message Text:", "\n".join(text_parts))
            else:
                # fallback: textual representation
                try:
                    print("🤖 Last Message:", last_msg.content.text)
                except Exception:
                    print("🤖 Last Message (raw):", last_msg.content)

        # Tool calls inside the message
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            tc = last_msg.tool_calls[0]
            name = getattr(tc.function, "name", str(tc.function))
            print(f"🛠️ Tool Called: {name}")
            # If there are args/inputs, print them
            try:
                args = getattr(tc, "args", None)
                if args:
                    print("   tool args:", args)
            except Exception:
                pass
            return

    # 2) Some event types return .text or .candidates
    if hasattr(last_event, "text") and last_event.text:
        print("🤖 Text:", last_event.text)

    if hasattr(last_event, "tool_calls") and last_event.tool_calls:
        print("🛠️ Tool Called:", last_event.tool_calls[0].function.name)
        return

    # 3) The Google GenAI responses sometimes contain .candidates with content
    if hasattr(last_event, "candidates") and last_event.candidates:
        try:
            cand = last_event.candidates[0]
            if hasattr(cand, "content") and cand.content:
                parts = getattr(cand.content, "parts", [])
                texts = []
                for p in parts:
                    try:
                        texts.append(p.text)
                    except Exception:
                        texts.append(str(p))
                print("🤖 Candidate text:", "\n".join(texts))
                return
        except Exception:
            pass

    print("⚠️ Response object found, but no known output or tool-calls detected. Dumping event for debugging:")
    try:
        print(repr(last_event)[:2000])
    except Exception:
        print(str(last_event)[:2000])


def run_tests(runner: Runner):
    # Test 1: Bio
    bio_thesis = "Please find recent articles discussing the ethical implications of human organoids."

    # Wrap into a types.Content object (has .role and .parts)
    bio_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=bio_thesis)],
    )
    runner.session_service.create_session(user_id="1", session_id="session_bio_1")
    # The runner.run returns an iterable/async generator of events.
    response_1_generator = runner.run(
        user_id="1",
        session_id="session_bio_1",
        new_message=bio_content
    )
    inspect_response(response_1_generator, "Biomedical Thesis")

    # Test 2: General
    general_thesis = "What are the major academic disagreements on the causes of the Renaissance?"

    general_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=general_thesis)],
    )
    runner.session_service.create_session(user_id="1", session_id="session_gen_1")
    response_2_generator = runner.run(
        user_id="1",
        session_id="session_gen_1",
        new_message=general_content
    )
    inspect_response(response_2_generator, "General Thesis")


if __name__ == "__main__":
    runner = initialize_runner()
    if runner:
        run_tests(runner)
