# main.py
import logging
import os
import asyncio
import uuid
from types import SimpleNamespace

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as gen_types

from app.core.agents import get_router_agent
from app.infrastructure.tools import PubMedTool, GoogleScholarTool


# -------------------------
# Logging / small utilities
# -------------------------
def cleanup_logs():
    for log_file in ["logger.log", "web.log", "tunnel.log"]:
        if os.path.exists(log_file):
            os.remove(log_file)


cleanup_logs()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ThesisAdvocator")


# -------------------------
# ADK Content helper
# -------------------------
def create_content_message(text: str) -> gen_types.Content:
    """Wraps text into ADK Content format."""
    return gen_types.Content(
        role="user",
        parts=[gen_types.Part.from_text(text=text)]
    )


# -------------------------
# Session helper (async)
# -------------------------
async def _create_session_async(runner, user_id, session_id, app_name):
    """Async helper for session creation."""
    await runner.session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )


# -------------------------
# Initialize system
# -------------------------
def initialize_system():
    """Sets up the App and Runner and returns (runner, app_name)."""
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


# -------------------------
# Main interactive loop
# -------------------------
def run_thesis_advocator():
    runner, app_name = initialize_system()
    if not runner:
        return

    # User & Session Setup
    USER_ID = "user_1"
    SESSION_ID = f"session_{uuid.uuid4().hex[:8]}"

    # Create Session (sync)
    asyncio.run(_create_session_async(runner, USER_ID, SESSION_ID, app_name))
    print(f"🎓 Thesis Advocator Initialized (Session: {SESSION_ID})")

    # Get initial thesis/idea
    thesis_input = input("\n📝 Enter your thesis/idea: ").strip()
    if not thesis_input:
        print("Empty input. Exiting.")
        return

    approved = False
    current_query = thesis_input

    # Keyword fallback list for deciding PubMed vs Scholar when router fails to call a tool
    bio_keywords = ["cell", "protein", "organoid", "genome", "stem", "cancer", "clinical", "disease", "therapy"]

    while not approved:
        print(f"\n🔎 Routing and Searching for: '{current_query}'...")

        # Build content and call runner.run with retry on runtime errors
        message = create_content_message(current_query)

        try:
            events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=message)
        except Exception as e:
            logger.exception("Runner failed during agent run: %s", e)
            retry = input("Agent run failed (network/async). Retry? (y/n): ").strip().lower()
            if retry in ("y", "yes"):
                continue
            else:
                print("Exiting due to runner error.")
                return

        # Collect textual output and detect tool calls (take the FIRST tool call seen)
        full_text_output = []
        tool_used = None
        tool_response_text = None

        for event in events:
            # Extract text parts (multiple ADK shapes handled)
            try:
                # Preferred: event.messages[*].content.parts[*].text
                if hasattr(event, "messages") and event.messages:
                    for m in event.messages:
                        if getattr(m, "content", None) and getattr(m.content, "parts", None):
                            for p in m.content.parts:
                                if getattr(p, "text", None):
                                    full_text_output.append(p.text)
                # Fallback: event.content.parts
                elif getattr(event, "content", None) and getattr(event.content, "parts", None):
                    for p in event.content.parts:
                        if getattr(p, "text", None):
                            full_text_output.append(p.text)
            except Exception:
                # ignore single-event parsing errors
                pass

            # Detect tool calls - we accept first tool call seen as canonical
            try:
                tcs = getattr(event, "tool_calls", None)
                if tcs:
                    seen = getattr(tcs[0].function, "name", None)
                    if not tool_used:
                        tool_used = seen
                    elif tool_used != seen:
                        logger.warning(
                            "Model attempted multiple tools in one response: first='%s' later='%s' — using first.",
                            tool_used, seen)
            except Exception:
                pass

        # Join text parts into final_response (may be empty if agent only issued a tool call)
        final_response = "\n".join(full_text_output).strip() or None

        # If agent DID NOT call a tool, do a deterministic fallback programmatic call
        if not tool_used:
            if any(k in current_query.lower() for k in bio_keywords):
                tool_used = "pubmed_tool"
                tool_response_text = PubMedTool(max_results=5).execute(current_query)
            else:
                tool_used = "google_scholar_tool"
                tool_response_text = GoogleScholarTool().execute(current_query)

            print("\n⚠️ Router didn't call tool — using fallback programmatic call.")
            final_response = tool_response_text

        # Display results to user
        print(f"\n🛠️ Tool Used: {tool_used}")
        print("-" * 40)
        print(final_response or "[No response text]")
        print("-" * 40)

        # Append tool/agent output into the session (so future agents can read it)
        try:
            content = gen_types.Content(role="assistant", parts=[gen_types.Part.from_text(text=str(final_response))])
            evt = SimpleNamespace(content=content)
            runner.session_service.append_event(user_id=USER_ID, session_id=SESSION_ID, event=evt)
        except Exception as e:
            logger.debug("Could not append event to session: %s", e)

        # ---------------------------
        # Human-in-the-loop: Approve/Refine/Quit
        # ---------------------------
        while True:
            choice = input("\n🤔 Are these results relevant? (y = yes / n = no, refine / q = quit): ").strip().lower()
            if choice in ("y", "yes"):
                approved = True
                print("\n✅ Evidence Approved.")
                break
            elif choice in ("q", "quit"):
                print("Exiting.")
                return
            elif choice in ("n", "no", "refine"):
                new_q = input("🔄 Enter refined keywords or query (or blank to re-run same query): ").strip()
                if new_q:
                    current_query = new_q
                # don't mark approved -> loop will rerun using current_query
                break
            else:
                print("Please type 'y' (yes), 'n' (no/refine) or 'q' (quit).")

    # End of research loop -> approved == True
    print("\n🎯 Finalizing — starting debate/synthesis phase (placeholder).")
    print(f"Context (original thesis): {thesis_input}")
    print(
        "Collected evidence appended to session; here you would call Pro/Con debator agents and synthesize their outputs.")
    # TODO: call your debator agents here, e.g. pass session context to them and append their results.


if __name__ == "__main__":
    run_thesis_advocator()
