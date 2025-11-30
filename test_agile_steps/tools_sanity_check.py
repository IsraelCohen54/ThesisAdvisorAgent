# sanity_check.py
import os
import inspect
import logging
import json

# ADK imports
from google.genai import types as gen_types
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# your agents/tools (make sure these import from your project)
from app.core.agents import get_talk_agent  # must return an LlmAgent router configured with AgentTools
from app.infrastructure.tools import GoogleScholarTool, PubMedTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sanity_check")

# Quick env checks
if not os.getenv("GEMINI_API_KEY"):
    logger.error("GEMINI_API_KEY missing in env. Set it in PyCharm run configuration.")
if not os.getenv("SERPAPI_API_KEY"):
    logger.warning("SERPAPI_API_KEY missing — Scholar evaluations will fail without it.")
if not os.getenv("NCBI_API_KEY"):
    logger.info("NCBI_API_KEY not set (ok for low-volume evaluations) — Entrez will still run but rate-limited.")


def ensure_session_sync(session_service, app_name, user_id, session_id):
    """
    Create session in a way that works whether create_session is sync or coroutine.
    This creates a temporary event loop if needed and closes it immediately.
    """
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


def safe_text_of(part):
    """Return textual representation for a content/part, safe vs None."""
    try:
        if getattr(part, "text", None):
            return part.text
        # function_response may exist
        fr = getattr(part, "function_response", None)
        if fr and getattr(fr, "response", None):
            # pretty-print JSON-like response if possible
            try:
                return json.dumps(fr.response, indent=2, ensure_ascii=False)
            except Exception:
                return str(fr.response)
    except Exception:
        pass
    return None


def pretty_print_results(source_name: str, results):
    """
    results: could be either a list of dicts (preferred) or a long string.
    Will print a short table: index | title | source | link
    """
    print(f"\n--- {source_name} (top {min(5, len(results) if results else 0)}) ---")
    if not results:
        print("[no results]")
        return

    # if results is a string, just print first 800 chars
    if isinstance(results, str):
        print(results[:800] + ("..." if len(results) > 800 else ""))
        return

    # assume list of dicts
    for i, r in enumerate(results[:5], start=1):
        title = r.get("title", "No Title")
        src = r.get("source", "") or r.get("authors", "")
        link = r.get("link", r.get("url", ""))
        print(f"{i}. {title}")
        if src:
            print(f"   Source: {src}")
        if link:
            print(f"   Link: {link}")
        snippet = r.get("snippet")
        if snippet:
            print(f"   Snippet: {snippet[:180]}{'...' if len(snippet) > 180 else ''}")
        print()



def print_event_debug(ev, index=None):
    """
    Print all interesting bits of an event to help debug ADK shapes:
      - messages -> content.parts -> text / function_response
      - content.parts -> text / function_response
      - tool_calls (with args)
      - raw event repr (short)
    """
    header = f"--- EVENT [{index}] ---" if index is not None else "--- EVENT ---"
    print(header)
    # quick repr
    try:
        print("repr:", repr(ev)[:1000])
    except Exception:
        pass

    # messages[*]
    try:
        if getattr(ev, "messages", None):
            print("Has ev.messages (count):", len(ev.messages))
            for mi, m in enumerate(ev.messages):
                print(f"  message[{mi}] role:", getattr(m, "role", None))
                if getattr(m, "tool_calls", None):
                    print("    message.tool_calls:", [getattr(tc.function, "name", "<no-name>") for tc in m.tool_calls])
                if getattr(m, "content", None) and getattr(m.content, "parts", None):
                    for pi, p in enumerate(m.content.parts):
                        pt = safe_text_of(p)
                        print(f"    msg.content.part[{pi}] text/funcresp: ", (pt[:500] if pt else None))
        elif getattr(ev, "content", None) and getattr(ev.content, "parts", None):
            print("Has ev.content.parts")
            for pi, p in enumerate(ev.content.parts):
                pt = safe_text_of(p)
                print(f"  content.part[{pi}] text/funcresp: ", (pt[:500] if pt else None))
    except Exception as e:
        print("Error while printing messages/content parts:", e)

    # top-level tool_calls
    try:
        tcs = getattr(ev, "tool_calls", None)
        if tcs:
            print("Top-level tool_calls detected:", len(tcs))
            for tc in tcs:
                try:
                    fname = getattr(tc.function, "name", "<no-name>")
                    print("  tool name:", fname)
                    if getattr(tc, "args", None):
                        print("   args:", tc.args)
                except Exception as e:
                    print("   error printing tool_call:", e)
    except Exception as e:
        print("Error inspecting tool_calls:", e)

    print("--- end event ---\n")


def main():
    router_agent = get_talk_agent()  # should return your LlmAgent router configured with AgentTools
    # Build a small app for the check
    app = App(name="sanity_app", root_agent=router_agent, resumability_config=ResumabilityConfig(is_resumable=True))
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service)

    USER = "test_user"
    SESSION = "sanity_session_1"

    # create session safely
    ensure_session_sync(runner.session_service, app.name, USER, SESSION)
    print("Session created.\n")

    # sanity query (biomedical)
    test_query = "immune system"

    from google.genai import types as gen_types
    content = gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=test_query)])

    print("\n--- Running router (expecting tool call to a researcher) ---\n")
    events = runner.run(user_id=USER, session_id=SESSION, new_message=content)

    last_event = None
    for idx, ev in enumerate(events):
        last_event = ev
        # print everything we see in each streaming event to aid debugging
        print_event_debug(ev, index=idx)

    if last_event is None:
        print("No events returned by runner.")
        return

    print("\n--- SUMMARY of last event ---")
    print_event_debug(last_event)

    # Try extracting textual parts from last_event for a quick human-readable check
    text_parts = []
    try:
        if hasattr(last_event, "messages") and last_event.messages:
            for m in last_event.messages:
                if getattr(m, "content", None) and getattr(m.content, "parts", None):
                    for p in m.content.parts:
                        if getattr(p, "text", None):
                            text_parts.append(p.text)
                        else:
                            fr = getattr(p, "function_response", None)
                            if fr and getattr(fr, "response", None):
                                text_parts.append(str(fr.response))
        elif getattr(last_event, "content", None) and getattr(last_event.content, "parts", None):
            for p in last_event.content.parts:
                if getattr(p, "text", None):
                    text_parts.append(p.text)
                else:
                    fr = getattr(p, "function_response", None)
                    if fr and getattr(fr, "response", None):
                        text_parts.append(str(fr.response))
    except Exception:
        pass

    if text_parts:
        joined = "\n".join(text_parts)
        print("\nRouter produced text/tool-response (summary):\n", joined[:4000])
        print(
            "\nIf this is a natural-language answer instead of a tool invocation, make the router instruction\nmore explicit ('CALL the tool named X' and provide examples).")
    else:
        print("\nNo textual parts found in last event (tool-call only or different event shape).")

    # Quick sanity: call tools directly (bypass agents) so you know they work locally
    print("\n--- Direct tool sanity checks ---")
    print("PubMed quick call (top 5):")
    pm_out = PubMedTool(max_results=5).execute("immune system")
    pretty_print_results("PubMed", pm_out.get("result") if isinstance(pm_out, dict) else pm_out)

    print("\nGoogle Scholar quick call (string result) — requires SERPAPI_API_KEY:")
    gs_out = GoogleScholarTool().execute("human organoids")
    pretty_print_results("Google Scholar", gs_out.get("result") if isinstance(gs_out, dict) else gs_out)


if __name__ == "__main__":
    main()
