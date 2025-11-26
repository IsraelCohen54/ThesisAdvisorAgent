# sanity_check.py
import os
import inspect
import logging
from types import SimpleNamespace

# ADK imports
from google.genai import types as gen_types
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# your agents/tools
from app.core.agents import get_router_agent  # this must return the router LlmAgent
from app.infrastructure.tools import GoogleScholarTool, PubMedTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sanity_check")

# Ensure GEMINI & SERPAPI keys are present (quick sanity)
if not os.getenv("GEMINI_API_KEY"):
    logger.error("GEMINI_API_KEY missing in env. Set it in PyCharm run configuration.")
if not os.getenv("SERPAPI_API_KEY"):
    logger.warning("SERPAPI_API_KEY missing — Scholar tests will fail without it.")


# small helper to create session whether create_session is sync or async
def ensure_session_sync(session_service, app_name, user_id, session_id):
    create_fn = getattr(session_service, "create_session", None)
    if create_fn is None:
        raise RuntimeError("session_service has no create_session")
    if inspect.iscoroutinefunction(create_fn):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(create_fn(app_name=app_name, user_id=user_id, session_id=session_id))
        finally:
            loop.close()
    else:
        create_fn(app_name=app_name, user_id=user_id, session_id=session_id)


def print_event_debug(ev):
    # Print message text parts
    try:
        if hasattr(ev, "messages") and ev.messages:
            for m in ev.messages:
                if getattr(m, "content", None) and getattr(m.content, "parts", None):
                    for p in m.content.parts:
                        if getattr(p, "text", None):
                            print("MSG PART >", p.text)
        elif getattr(ev, "content", None) and getattr(ev.content, "parts", None):
            for p in ev.content.parts:
                if getattr(p, "text", None):
                    print("CONTENT PART >", p.text)
    except Exception as e:
        print("Failed printing text parts:", e)

    # Print any tool_calls attached to the event
    tcs = getattr(ev, "tool_calls", None)
    if tcs:
        for tc in tcs:
            try:
                name = getattr(tc.function, "name", "<no-name>")
                print("TOOL CALL DETECTED ->", name)
                # print args if present
                if getattr(tc, "args", None):
                    print("  args:", tc.args)
            except Exception as e:
                print("tool_call inspect error:", e)


def main():
    router_agent = get_router_agent()  # returns an LlmAgent router configured with AgentTools
    app = App(name="sanity_app", root_agent=router_agent, resumability_config=ResumabilityConfig(is_resumable=True))
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service)

    USER = "test_user"
    SESSION = "sanity_session_1"

    # create session safely
    ensure_session_sync(runner.session_service, app.name, USER, SESSION)
    print("Session created.")

    # Make a test query that is clearly biomedical (should route to PubMedResearcher)
    test_query = "immune system"

    content = gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=test_query)])

    print("\n--- Running router (expecting tool call to a researcher) ---\n")
    events = runner.run(user_id=USER, session_id=SESSION, new_message=content)

    last_event = None
    for ev in events:
        last_event = ev
        # print everything we see in each streaming event to aid debugging
        print_event_debug(ev)

    if last_event is None:
        print("No events returned by runner.")
        return

    # If the router called a tool, runner usually returns additional events with the tool/run outputs.
    # Check last_event for any tool_calls or message text
    print("\n--- SUMMARY of last event ---")
    print_event_debug(last_event)

    # If router didn't call a tool, show last textual output for diagnosis
    # (helps you tune router instruction)
    text_parts = []
    try:
        if hasattr(last_event, "messages") and last_event.messages:
            for m in last_event.messages:
                if getattr(m, "content", None) and getattr(m.content, "parts", None):
                    for p in m.content.parts:
                        if getattr(p, "text", None):
                            text_parts.append(p.text)
    except Exception:
        pass

    if text_parts:
        joined = "\n".join(text_parts)
        print("\nRouter produced text (no tool call detected?):\n", joined[:2000])
        print(
            "\nIf this is a natural-language answer instead of a tool invocation, try making the router instruction "
            "more explicit and add 'CALL the tool' examples.")
    else:
        print("\nNo textual parts found in last event (tool-call only or different event shape).")

    # Quick sanity: call tools directly (bypass agents) so you know they work
    print("\n--- Direct tool sanity checks ---")
    print("PubMed quick call (top 1):")
    pm = PubMedTool(max_results=5).execute("immune system")
    print(pm)

    print("\nGoogle Scholar quick call (string result) — requires SERPAPI_API_KEY:")
    gs = GoogleScholarTool().execute("human organoids")
    print(gs)


if __name__ == "__main__":
    main()
