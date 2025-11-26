# routing_logic (check usage of tools + runners and user query)
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import logging
import os
from app.core.agents import get_router_agent, ScholarResearcher, PubMedResearcher  # from agents.py
from google.adk.tools.agent_tool import AgentTool
import asyncio
import inspect
from google.genai import types
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.runners import InMemoryRunner
import json
import requests
import subprocess
import time
import uuid
from google.adk.agents import LlmAgent

from app.infrastructure.tools import GoogleScholarTool, PubMedTool

"""from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)
from google.adk.a2a.utils.agent_to_a2a import to_a2a
"""
from google.genai import types as gen_types


def ensure_session(runner, app_name: str, user_id: str, session_id: str):
    """
    Create a session safely. Works whether session_service.create_session is sync or coroutine.
    Uses a fresh event loop for coroutine invocation to avoid closing the main loop.
    """
    create_fn = getattr(runner.session_service, "create_session", None)
    if create_fn is None:
        raise RuntimeError("Runner has no session_service.create_session")

    if inspect.iscoroutinefunction(create_fn):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(create_fn(app_name=app_name, user_id=user_id, session_id=session_id))
        finally:
            loop.close()
    else:
        # sync function
        create_fn(app_name=app_name, user_id=user_id, session_id=session_id)


# New helper function
def create_content_message(text: str) -> gen_types.Content:
    """Wraps a string into the required Content object for the Runner."""
    return gen_types.Content(
        role="user",
        parts=[gen_types.Part.from_text(text=text)]
    )


def cleanup_logs():
    for log_file in ["logger.log", "web.log", "tunnel.log"]:
        if os.path.exists(log_file):
            os.remove(log_file)


cleanup_logs()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ThesisAdvocator")


def show_python_code_and_result(response):
    for i in range(len(response)):
        # Check if the response contains a valid function call result from the code executor
        if (
                (response[i].content.parts)
                and (response[i].content.parts[0])
                and (response[i].content.parts[0].function_response)
                and (response[i].content.parts[0].function_response.response)
        ):
            response_code = response[i].content.parts[0].function_response.response
            if "result" in response_code and response_code["result"] != "```":
                if "tool_code" in response_code["result"]:
                    print(
                        "Generated Python Code >> ",
                        response_code["result"].replace("tool_code", ""),
                    )
                else:
                    print("Generated Python Response >> ", response_code["result"])


def initialize_runner():
    # Check for simplified key
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("FATAL: GEMINI_API_KEY not found.")
        return None

    # 1. Create Session
    session_service = InMemorySessionService()

    # 2. Create Agent (Clean init)
    advocator_agent = get_router_agent()

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


def route_keywords(text: str) -> str:
    """Simple fallback routing if LLM router didn't call a tool."""
    bio_keywords = ["cell", "protein", "organoid", "genome", "stem", "cancer", "clinical", "disease", "therapy", "biomedical", "immuno"]
    t = text.lower()
    return "PubMedResearcher" if any(k in t for k in bio_keywords) else "ScholarResearcher"

def print_agent_response_events(events):
    """Print agent's text responses and check for tool calls from event stream."""

    final_output_text = []
    tool_called_name = None

    # Iterate through all events yielded by the runner
    for event in events:
        content = getattr(event, "content", None)

        # 1. Capture Text Output
        if content:
            parts = getattr(content, "parts", []) or []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    final_output_text.append(text)

        # 2. Capture Tool Calls (The "Did it route correctly?" check)
        if hasattr(event, 'tool_calls') and event.tool_calls:
            # The agent decided to call a tool (the correct step for a router)
            tool_called_name = event.tool_calls[0].function.name

    print(f"\n--- 🔑 RESULT SUMMARY ---")
    if tool_called_name:
        print(f"🛠️ Tool Called: **{tool_called_name}** (SUCCESS)")
    else:
        print("⚠️ Tool Status: No external tool was triggered.")

    if final_output_text:
        # Concatenate and print the LLM's final response (if any)
        print("\n🤖 Final Agent Output:")
        print("--------------------")
        print(" ".join(final_output_text))
    else:
        print("🤖 Final Agent Output: [None]")


async def _create_session_async(runner, user_id, session_id, app_name):
    """Helper coroutine to call the async session creation."""
    await runner.session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )

def format_pubmed_results(results):
    if not results:
        return "No PubMed results."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title')} ({r.get('pubdate')})")
        lines.append(f"   Authors: {r.get('authors')}")
        lines.append(f"   PMID: {r.get('pmid')} | URL: {r.get('url')}")
        if r.get('snippet'):
            lines.append(f"   Snippet: {r.get('snippet')[:300]}")
        lines.append("")
    return "\n".join(lines)

def run_sub_agent_and_handle_refinement(runner, user_id, session_id, thesis_text):
    """Simple sub-agent runner: ask the user for clarification and re-run a researcher."""
    more = input("Please give a refined thesis phrase or keywords (or press Enter to abort): ").strip()
    if not more:
        print("No refinement. Aborting refinement stage.")
        return
    refined = thesis_text + " " + more
    # Programmatic simple route again (could be more advanced)
    chosen = route_keywords(refined)
    if chosen == "PubMedResearcher":
        resp = PubMedTool(max_results=5).execute(refined)
        text = format_pubmed_results(resp.get("result", []))
    else:
        resp = GoogleScholarTool().execute(refined)
        text = resp.get("result")
    print("\nRefined search results:\n")
    print(text)
    # Optionally append and loop, or ask user again...

def create_sessions_sync(runner, user_id, session_id_1, session_id_2, app_name):
    """
    Synchronously creates sessions by running the coroutines.
    """
    # Use asyncio.run to execute the coroutines, stopping the RuntimeWarning
    # and ensuring the sessions are actually created.
    asyncio.run(_create_session_async(runner, user_id, session_id_1, app_name))
    asyncio.run(_create_session_async(runner, user_id, session_id_2, app_name))


def run_tests(runner: Runner):
    USER_SESSION = "debate_session_123"
    USER_ID = "1"
    APP_NAME = "ThesisAdvocatorApp"

    ensure_session(runner, APP_NAME, USER_ID, USER_SESSION)

    # Build router as root agent in app (if you built app differently, ensure router is used)
    # If your App already uses root_agent from initialize_runner, skip and just use runner.run as below.

    thesis_text = "Please find recent articles discussing the ethical implications of human organoids."

    # Make content
    query_content = create_content_message(thesis_text)

    # Run router; router should call an AgentTool (ScholarResearcher or PubMedResearcher)
    events = runner.run(user_id=USER_ID, session_id=USER_SESSION, new_message=query_content)

    last_event = None
    for ev in events:
        last_event = ev

    # --- inspect for a tool_call (AgentTool invocation) ---
    chosen_tool_name = None
    tool_result_text = None

    if last_event is None:
        print("⚠️ No events returned from router.")
        return

    # 1) If ADK message with messages array:
    if hasattr(last_event, "messages") and last_event.messages:
        last_msg = last_event.messages[-1]
        # If the model properly invoked the AgentTool, ADK will surface a tool_call
        if getattr(last_msg, "tool_calls", None):
            tc = last_msg.tool_calls[0]
            # tool function name corresponds to the AgentTool.name
            chosen_tool_name = getattr(tc.function, "name", None)
            # tool response content (the researcher agent's output) might be in tc.tool_results or in subsequent messages.
            # We try to read the returned message(s) from last_event.messages (tool-run will usually produce an assistant message)
            parts = []
            for m in last_event.messages:
                try:
                    cp = getattr(m, "content", None)
                    if cp and getattr(cp, "parts", None):
                        for p in cp.parts:
                            if getattr(p, "text", None):
                                parts.append(p.text)
                except Exception:
                    continue
            tool_result_text = "\n".join(parts).strip() or None

    # 2) Fallback: check event.tool_calls attribute on the raw event (some ADK versions put tool_calls there)
    if not chosen_tool_name and getattr(last_event, "tool_calls", None):
        tc = last_event.tool_calls[0]
        chosen_tool_name = getattr(tc.function, "name", None)
        # try to grab text from content
        if getattr(last_event, "content", None) and getattr(last_event.content, "parts", None):
            tool_result_text = " ".join(p.text for p in last_event.content.parts if getattr(p, "text", None))

    # 3) If still no tool chosen, use keyword fallback and call tool programmatically
    if not chosen_tool_name:
        print("Router didn't call a tool. Applying keyword fallback.")
        chosen_tool_name = route_keywords(thesis_text)
        print("Fallback chose:", chosen_tool_name)
        # Programmatically call the underlying tool implementation:
        if chosen_tool_name == "PubMedResearcher":
            tool_resp = PubMedTool(max_results=5).execute(thesis_text)
            tool_result_text = format_pubmed_results(tool_resp.get("result", []))
        else:
            tool_resp = GoogleScholarTool().execute(thesis_text)
            tool_result_text = tool_resp.get("result")

    # --- Present results to user ---
    print("\n--- Results from researcher:", chosen_tool_name, "---\n")
    print(tool_result_text or "[No textual result captured]")

    # Append tool result to session so next agents can read it
    try:
        from types import SimpleNamespace
        from google.genai import types as gen_types
        content = gen_types.Content(role="assistant", parts=[gen_types.Part.from_text(text=str(tool_result_text))])
        evt = SimpleNamespace(content=content)
        runner.session_service.append_event(user_id=USER_ID, session_id=USER_SESSION, event=evt)
    except Exception as e:
        print("Warning: cannot append tool result to session:", e)

    # Ask for confirmation (stdin)
    proceed = input("\nAre these results similar to your thesis idea? (y/n): ").strip().lower()
    if proceed not in ("y", "yes"):
        # Launch a sub-agent that takes ownership and performs iterative search/refinement.
        print("Launching SubAgent to refine thesis and search...")
        run_sub_agent_and_handle_refinement(runner, USER_ID, USER_SESSION, thesis_text)
        return

    # Otherwise proceed to debators as we discussed earlier
    print("Proceeding to multi-agent debate (pro/con synthesis)...")


if __name__ == "__main__":
    runner = initialize_runner()
    if runner:
        run_tests(runner)
