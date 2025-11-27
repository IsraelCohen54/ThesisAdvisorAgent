# main.py  -- REPLACE your existing file with this
import logging
import os
import uuid
import inspect
import json
from types import SimpleNamespace
import asyncio

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as gen_types

from app.core.agents import get_talk_agent
from app.infrastructure.tools import PubMedTool, GoogleScholarTool

# ---------------- Logging: filter the App-name-mismatch warning ----------------
class AppNameMismatchFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Filter out the specific ADK warning text that is harmless but noisy
        msg = record.getMessage()
        if "App name mismatch detected" in msg:
            return False
        return True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# reduce very noisy ADK/httpx low-level logs while dev'ing
logging.getLogger("google_adk").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("ThesisAdvocator")
logger.addFilter(AppNameMismatchFilter())


# ---------------- Helpers ----------------

def create_content_message(text: str) -> gen_types.Content:
    return gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=text)])

def ensure_session_sync(session_service, app_name: str, user_id: str, session_id: str):
    """
    Create session whether create_session is sync or async. Avoids calling asyncio.run()
    while ADK threads may be active.
    """
    create_fn = getattr(session_service, "create_session", None)
    if create_fn is None:
        raise RuntimeError("session_service has no create_session")
    if inspect.iscoroutinefunction(create_fn):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(create_fn(app_name=app_name, user_id=user_id, session_id=session_id))
        finally:
            loop.close()
    else:
        create_fn(app_name=app_name, user_id=user_id, session_id=session_id)


def _normalize_tool_payload(raw):
    """
    Accepts many shapes and returns:
      - Python list of dicts (preferred)
      - or a string fallback
    Handles:
      - {"result": [...]}
      - list([...])
      - JSON string of list/dict
      - plain string
    """
    if raw is None:
        return None
    # If it's a dict that contains "result"
    if isinstance(raw, dict) and "result" in raw:
        payload = raw["result"]
        # sometimes result is JSON string
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
                return parsed
            except Exception:
                return payload
        return payload
    # If it's already a list
    if isinstance(raw, list):
        return raw
    # If it's a JSON-encoded string
    if isinstance(raw, str):
        s = raw.strip()
        if (s.startswith("{") or s.startswith("[")):
            try:
                parsed = json.loads(s)
                return parsed
            except Exception:
                # not JSON or cannot parse -> return original string
                return raw
        # not JSON -> just return the string
        return raw
    # unknown type -> string-ify
    return str(raw)


def format_results_for_display(tool_payload) -> str:
    """
    Given tool_payload (list of dicts, a dict, or string), return a pretty multi-line string.
    Each item: Title / Authors / Source / Link / snippet
    """
    payload = _normalize_tool_payload(tool_payload)

    # if payload is None or empty
    if not payload:
        return "I could not find any results for your query."

    # if payload is a string -> return as-is (but keep short)
    if isinstance(payload, str):
        return payload if len(payload) < 4000 else payload[:4000] + "..."

    # If it's a dict (not expected now after normalization), stringify politely
    if isinstance(payload, dict):
        # try to extract list under 'result' or 'results'
        candidate = payload.get("result") or payload.get("results")
        if candidate:
            payload = candidate
        else:
            # just pretty print keys
            return json.dumps(payload, indent=2)[:4000]

    # Now we expect a list of dicts
    if isinstance(payload, list):
        lines = []
        for i, r in enumerate(payload[:10], start=1):  # show up to 10 but usually 5
            # r may be dict or string
            if isinstance(r, str):
                # string entry
                lines.append(f"{i}. {r}")
                lines.append("")
                continue

            title = r.get("title") or r.get("Title") or r.get("TI") or r.get("name") or "No title"
            authors = r.get("authors") or r.get("Authors") or r.get("AU")
            source = r.get("source") or r.get("journal") or r.get("SOURCE")
            link = r.get("link") or r.get("url") or r.get("Link")
            pmid = r.get("pmid") or r.get("PMID")
            snippet = r.get("snippet") or r.get("abstract") or r.get("SNIPPET") or ""

            # Build display lines
            # Bold-like title using ** to be visible in console
            lines.append(f"{i}. {title}")
            meta_parts = []
            if authors:
                meta_parts.append(f"Authors: {authors if isinstance(authors,str) else ', '.join(authors[:3])}")
            if source:
                meta_parts.append(f"Source: {source}")
            if pmid:
                meta_parts.append(f"PMID: {pmid}")
            if meta_parts:
                lines.append("   " + " | ".join(meta_parts))
            if link:
                lines.append(f"   Link: {link}")
            if snippet:
                s = snippet.strip().replace("\n", " ")
                lines.append(f"   {s[:260]}{'...' if len(s) > 260 else ''}")
            lines.append("")  # blank line
        return "\n".join(lines)

    # Fallback
    return str(payload)


# ---------------- initialize runner & app ----------------

def initialize_system():
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("FATAL: GEMINI_API_KEY not found.")
        return None, None

    talk_agent = get_talk_agent()
    app = App(name="ThesisAdvocatorApp", root_agent=talk_agent,
              resumability_config=ResumabilityConfig(is_resumable=True))
    runner = Runner(app=app, session_service=InMemorySessionService())
    return runner, app.name


# ---------------- Debater prompts: produce feasibility-focused pro/con --------------

def run_debaters(thesis_text: str, evidence_text: str):
    """
    Run two synchronous Gemini calls (Pro/Con). Each should return:
    - 3 concise reasons (pro/con)
    - a feasibility score (1-5) with short rationale
    """
    from google.adk.models.google_llm import Gemini
    model = Gemini(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
    pro_prompt = (
        f"You are PRO debater evaluating whether the student should pursue this thesis:\n\n"
        f"Thesis: {thesis_text}\n\n"
        f"Evidence (closest articles found):\n{evidence_text}\n\n"
        "Output a JSON object with keys: 'reasons' (list of 3 concise pro reasons), "
        "'feasibility' (integer 1..5), 'feasibility_rationale' (one short sentence), "
        "and 'summary' (one short paragraph). Keep it factual and cite which evidence items support each reason if possible."
    )
    con_prompt = (
        f"You are CON debater evaluating whether the student should pursue this thesis:\n\n"
        f"Thesis: {thesis_text}\n\n"
        f"Evidence (closest articles found):\n{evidence_text}\n\n"
        "Output a JSON object with keys: 'reasons' (list of 3 concise con reasons), "
        "'feasibility' (integer 1..5 - how risky/impractical it is), 'feasibility_rationale' (one short sentence), "
        "and 'summary' (one short paragraph). Keep it factual and cite which evidence items support each reason if possible."
    )

    try:
        pro_resp = model.api_client.models.generate_content(model="gemini-2.5-flash", contents=pro_prompt)
        con_resp = model.api_client.models.generate_content(model="gemini-2.5-flash", contents=con_prompt)
    except Exception as e:
        logger.exception("Debator call failed: %s", e)
        print("Debator call failed:", e)
        return None

    pro_text = getattr(pro_resp, "text", str(pro_resp)[:2000])
    con_text = getattr(con_resp, "text", str(con_resp)[:2000])

    # Try to parse JSON outputs; if fail, just show text.
    try:
        pro_json = json.loads(pro_text)
    except Exception:
        pro_json = {"raw": pro_text}

    try:
        con_json = json.loads(con_text)
    except Exception:
        con_json = {"raw": con_text}

    # Print nicely
    print("\n--- PRO Debator ---")
    if "reasons" in pro_json:
        for i, r in enumerate(pro_json.get("reasons", []), start=1):
            print(f"{i}. {r}")
        print(f"Feasibility (1-5): {pro_json.get('feasibility')}")
        print("Rationale:", pro_json.get("feasibility_rationale", ""))
        print("Summary:", pro_json.get("summary", ""))
    else:
        print(pro_text)

    print("\n--- CON Debator ---")
    if "reasons" in con_json:
        for i, r in enumerate(con_json.get("reasons", []), start=1):
            print(f"{i}. {r}")
        print(f"Feasibility (1-5): {con_json.get('feasibility')}")
        print("Rationale:", con_json.get("feasibility_rationale", ""))
        print("Summary:", con_json.get("summary", ""))
    else:
        print(con_text)

    # Synthesis
    try:
        synth_prompt = (
            "Synthesize the PRO and CON points into a neutral recommendation (1 short paragraph). "
            f"Pro: {json.dumps(pro_json)}\nCon: {json.dumps(con_json)}"
        )
        synth_resp = model.api_client.models.generate_content(model="gemini-2.5-flash", contents=synth_prompt)
        synth_text = getattr(synth_resp, "text", str(synth_resp)[:2000])
        print("\n--- SYNTHESIS ---\n", synth_text)
    except Exception as e:
        logger.exception("Synthesis failed: %s", e)

    return {"pro": pro_json, "con": con_json}


# ---------------- Main interactive loop ----------------

def run_thesis_advocator():
    runner, app_name = initialize_system()
    if not runner:
        return

    USER_ID = "user_1"
    session_service = runner.session_service

    print("🎓 Thesis Advocator — interactive (TalkAgent as root)")

    thesis_input = input("\n📝 Enter your thesis/idea: ").strip()
    if not thesis_input:
        print("Empty input — exiting.")
        return

    # IMPORTANT: create only one session at the start and reuse it for all refine/loops.
    SESSION_ID = f"session_{uuid.uuid4().hex[:8]}"
    ensure_session_sync(session_service, app_name, USER_ID, SESSION_ID)
    print(f"Using session: {SESSION_ID}")

    while True:
        print(f"\n🔎 Searching (session={SESSION_ID}) for: {thesis_input}")

        content = create_content_message(thesis_input)
        try:
            events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
        except Exception as e:
            logger.exception("Runner.run() failed: %s", e)
            print("Agent run failed:", e)
            return

        # Collect text and detect function responses
        collected_text_parts = []
        tool_payload = None
        tool_used = None

        for ev in events:
            # textual parts
            if getattr(ev, "messages", None):
                for m in ev.messages:
                    if getattr(m, "content", None) and getattr(m.content, "parts", None):
                        for p in m.content.parts:
                            if getattr(p, "text", None):
                                collected_text_parts.append(p.text)
                            # check for function_response stored in a part
                            if getattr(p, "function_response", None):
                                fr = p.function_response
                                # fr.response may already be a dict
                                resp = getattr(fr, "response", None)
                                tool_payload = resp or tool_payload

            elif getattr(ev, "content", None) and getattr(ev.content, "parts", None):
                for p in ev.content.parts:
                    if getattr(p, "text", None):
                        collected_text_parts.append(p.text)
                    if getattr(p, "function_response", None):
                        fr = p.function_response
                        resp = getattr(fr, "response", None)
                        tool_payload = resp or tool_payload

            # tool_calls metadata (which tool was called)
            if getattr(ev, "tool_calls", None):
                try:
                    tc = ev.tool_calls[0]
                    tool_used = getattr(tc.function, "name", None)
                except Exception:
                    pass

        # Prefer structured tool_payload if present
        if tool_payload:
            display = format_results_for_display(tool_payload)
        else:
            # if agent returned text parts, join them
            text_joined = "\n".join(collected_text_parts).strip()
            display = text_joined or "No textual output from agent."

        # Fallback to local tools if neither tool_payload nor agent text useful
        if (not tool_payload) and (not collected_text_parts):
            print("⚠️ Router didn't call a tool — using local fallback.")
            bio_keywords = ["cell", "protein", "organoid", "genome", "stem", "cancer", "clinical", "disease", "therapy"]
            if any(k in thesis_input.lower() for k in bio_keywords):
                raw = PubMedTool(max_results=5).execute(thesis_input)
            else:
                raw = GoogleScholarTool().execute(thesis_input)
            # normalize result: our tools sometimes return string or dict/list
            display = format_results_for_display(raw)

        # Present results nicely
        print("\n🛠️ Tool Used:", tool_used or "[unknown]")
        print("-" * 70)
        print(display)
        print("-" * 70)

        # Append result into session as assistant content (so debaters can read later)
        try:
            cont = gen_types.Content(role="assistant", parts=[gen_types.Part.from_text(text=str(display))])
            evt = SimpleNamespace(content=cont)
            session_service.append_event(user_id=USER_ID, session_id=SESSION_ID, event=evt)
        except Exception as e:
            logger.debug("Could not append event to session: %s", e)

        # Human in the loop question
        print("\nQuestion: Are one of these articles exactly like your thesis idea?")
        answer = input("Type (y = yes / n = no / r = refine / q = quit): ").strip().lower()

        if answer in ("y", "yes"):
            print("\n✅ Looks like an exact or extremely close match exists.")
            follow = input("Type (suggest = propose variants / refine = refine query / continue = run debate anyway / q = quit): ").strip().lower()
            if follow == "suggest":
                new = input("Enter a new thesis idea you'd like to explore (blank to quit): ").strip()
                if not new:
                    print("No new thesis given. Exiting.")
                    return
                thesis_input = new
                continue
            elif follow == "refine":
                refined = input("Enter refined thesis: ").strip()
                if not refined:
                    print("No refinement entered. Exiting.")
                    return
                thesis_input = refined
                continue
            elif follow == "continue":
                run_debaters(thesis_text=thesis_input, evidence_text=display)
                return
            else:
                print("Exiting.")
                return

        elif answer in ("n", "no"):
            print("\nGreat — proceeding to debate.")
            run_debaters(thesis_text=thesis_input, evidence_text=display)
            return

        elif answer in ("r", "refine"):
            refined = input("Enter refined keywords or query (or blank to reuse same): ").strip()
            if refined:
                thesis_input = refined
            else:
                print("Re-running same query.")
            # **do not recreate session** - reuse same session_id to avoid async race conditions
            continue

        elif answer in ("q", "quit"):
            print("Goodbye.")
            return

        else:
            print("Unrecognized input — treating as refine.")
            refined = input("Enter refined keywords or query: ").strip()
            if refined:
                thesis_input = refined
            else:
                print("Re-running same query.")
            continue


if __name__ == "__main__":
    run_thesis_advocator()
