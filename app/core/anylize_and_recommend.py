# app/core/anylize_and_recommend.py
import ast
import json
import time
import asyncio
from typing import Any, List, Callable, Dict, Optional
from app.config.settings import CORE_MODEL, logger
from google.genai import types as gen_types
from app.core.agents import DEBATE_SEARCH_TOOLS
from google.adk.runners import Runner

# ---------------------------
# Robust model call helper(s)
# ---------------------------
def call_model(model_obj, model_name, contents):
    """
    Try to call generate_content robustly: try with `contents` as-is,
    but if the SDK expects a string prompt, fall back to str(contents).
    Returns the model response object or raises the final exception.
    """
    try:
        return model_obj.api_client.models.generate_content(model=model_name, contents=contents)
    except TypeError:
        # Fallback: send joined plain-text prompt if the SDK expects a string
        try:
            if isinstance(contents, (list, tuple)):
                pieces = []
                for c in contents:
                    parts = getattr(c, "parts", None) or (c.get("parts") if isinstance(c, dict) else None)
                    if parts:
                        for p in parts:
                            txt = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
                            if txt:
                                pieces.append(str(txt))
                fallback_prompt = "\n\n".join(pieces) if pieces else str(contents)
            else:
                fallback_prompt = str(contents)
            return model_obj.api_client.models.generate_content(model=model_name, contents=fallback_prompt)
        except Exception:
            # Reraise the original TypeError if fallback failed to make debugging easier
            raise

def safe_get_text(resp) -> str:
    """
    Return the most-likely human text from a response object:
    prefer resp.text, then try candidates -> content -> parts -> text, else str(resp).
    """
    txt = getattr(resp, "text", None)
    if t:
        return txt
    candidates = getattr(resp, "candidates", None) or (resp.get("candidates") if isinstance(resp, dict) else None)
    if candidates:
        for cand in candidates:
            content = getattr(cand, "content", None) or (cand.get("content") if isinstance(cand, dict) else None)
            parts = content.get("parts") if isinstance(content, dict) else getattr(content, "parts", None)
            if parts:
                texts = []
                for p in parts:
                    texts.append(getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None) or str(p))
                return "\n".join([str(x) for x in texts if x])
    # last resort
    return str(resp)


# ---------------------------------------------------------------------
# Utilities: map tool names to callables (from your FunctionTool wrappers)
# ---------------------------------------------------------------------
TOOL_MAP: Dict[str, Callable[..., Any]] = {}
# Removed: AGENT_TOOL_MAP is no longer needed

for t in DEBATE_SEARCH_TOOLS:
    nm = getattr(t, "name", None)
    # DEBATE_SEARCH_TOOLS only contains FunctionTools
    if nm and hasattr(t, 'func'):
        TOOL_MAP[nm] = getattr(t, 'func') # FunctionTool


# ---------------------------------------------------------------------
# Small helper: inspect model resp object for function_call parts robustly
# ---------------------------------------------------------------------
def extract_function_call_from_resp(resp: Any) -> Optional[Dict[str, Any]]:
    """
    Try multiple likely locations in the response object to find a function_call part.
    Returns dict: {"name": str, "arguments": dict_or_str} or None if not found.
    """
    try:
        candidates = getattr(resp, "candidates", None)
        if candidates:
            for cand in candidates:
                content = getattr(cand, "content", None) or (cand.get("content") if isinstance(cand, dict) else None)
                if content:
                    parts = content.get("parts") if isinstance(content, dict) else getattr(content, "parts", None)
                    if parts:
                        for p in parts:
                            if isinstance(p, dict):
                                fc = p.get("function_call") or p.get("functionCall") or p.get("tool_call")
                                if fc:
                                    return {"name": fc.get("name"), "arguments": fc.get("args") or fc.get("arguments")}
                            else:
                                fc = getattr(p, "function_call", None) or getattr(p, "functionCall", None)
                                if fc:
                                    name = getattr(fc, "name", None)
                                    args = getattr(fc, "args", None) or getattr(fc, "arguments", None)
                                    return {"name": name, "arguments": args}
        # Fallback: string/repr search (last resort)
        text_repr = str(resp)
        if "function_call" in text_repr or '"function_call"' in text_repr:
            try:
                start = text_repr.index("{", text_repr.index("function_call"))
                j = json.loads(text_repr[start:])
                fc = j.get("function_call") or j.get("functionCall")
                if fc:
                    return {"name": fc.get("name"), "arguments": fc.get("args") or fc.get("arguments")}
            except Exception:
                pass
        return None
    except Exception as e:
        logger.debug("[extract_function_call_from_resp] unexpected parsing error: %s", e)
        return None


# ---------------------------------------------------------------------
# Host-side "function call" orchestration
# ---------------------------------------------------------------------
def run_tool_and_get_result(tool_name: str, raw_args: Any) -> Any:
    """
    Execute the FunctionTool identified by tool_name.
    raw_args may be a dict, a JSON string, or simple text.
    """
    fn = TOOL_MAP.get(tool_name)
    if not fn:
        # This will now only catch FunctionTools that are missing,
        # since AgentTool logic was removed.
        return {"error": f"Tool '{tool_name}' not found on host."}

    args = raw_args
    if isinstance(raw_args, str):
        raw = raw_args.strip()
        try:
            if raw.startswith("{") or raw.startswith("["):
                args = json.loads(raw)
        except Exception:
            args = raw_args

    try:
        if isinstance(args, dict):
            q = args.get("query") or args.get("q") or args.get("text") or args.get("input")
            if q is not None:
                return fn(q)
            try:
                return fn(**args)
            except Exception:
                return fn(str(args))
        else:
            return fn(str(args))
    except Exception as e:
        logger.exception("[run_tool_and_get_result] tool execution failed for %s: %s", tool_name, e)
        return {"error": str(e)}


# -----------------------
# Debate Agent & Judge
# -----------------------
class ContextAwareDebateAgent:
    def __init__(
        self,
        name: str,
        stance: str,
        thesis_text: str,
        formatted_references: str,
        criteria_context: str,
        tools: List[Any],
        # Removed unused ADK parameters: runner, user_id, session_id
    ):
        self.name = name
        self.stance = stance.upper()  # "PRO" or "CON"

        self.base_instruction = (
            f"You are {name}. Stance: {self.stance}. Your goal is to be highly persuasive.\n"
            f"Thesis Topic: {thesis_text}\n"
            f"Evaluation Criteria chosen by user: {criteria_context}\n\n"
            f"{formatted_references}\n\n"
            "GENERAL GUIDELINES:\n"
            "1. Be persuasive based on the criteria chosen.\n"
            "2. Use specific details from the provided references to support your point.\n"
            "3. Maintain professional, concise, and logical consistency. **Be highly detailed and elaborate when needed**.\n"
            "4. **CRITICAL:** You have access to only two **academic search tools: 'google_scholar_execute' and 'pubmed_execute'**. You can use only **ONE** academic tool per round when searching is explicitly permitted.\n"
            " **Round 1 (Opening):** Use ONLY the references provided above, no search tool use is allowed. If you find any supporting data from reference, say that clearly.\n"
            " **Rounds 3 and 5 (Strengthening):** You are explicitly permitted to use one academic search tool ('google_scholar_execute' or 'pubmed_execute') to find new research literature to strengthen your case.\n"
            " **Rounds 2 and 4 (Rebuttal):** Focus on refuting the opponent using existing context; search is discouraged unless absolutely necessary for a counter-claim that requires academic evidence.\n"
            " Clarification: Somewhat ambiguous in thesis isn't a claim,"
            " assume user can use method available to humanity and has the capabilities,"
            " focus on thesis idea and potential, not on the \"not good enough\" wording or not elaborate research method."
        )

        self.model = CORE_MODEL
        self.tools = tools

        # Keep only base + last response to limit context growth
        self.history: List[str] = [self.base_instruction]

    def _build_call_contents(self, last_response: str, context_prompt: str) -> List[gen_types.Content]:
        """Build the genai contents list: system (base instruction) + user (last + current)"""
        history_text = last_response if last_response else "No previous response."
        user_text = (
            f"{self.base_instruction}\n\n"
            f"PREVIOUS (last) RESPONSE:\n{history_text}\n\n"
            f"CURRENT INSTRUCTION:\n{context_prompt}"
        )
        return [
            gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=user_text)])
        ]

    def argue(self, context_prompt: str) -> str:
        """
        Generates an argument. This function:
        - Sends base + last response + current instruction to the model
        - If the model returns a function_call, the host runs the tool and re-invokes
          the model including the tool result so the model can incorporate it.
        - Returns the final model text output.
        """
        last_response = self.history[1] if len(self.history) > 1 else ""

        contents = self._build_call_contents(last_response, context_prompt)

        model_name = getattr(self.model, "model", "gemini-2.5-flash")

        # 1) initial call
        resp = call_model(self.model, model_name, contents)

        # 2) inspect for function_call
        fc = extract_function_call_from_resp(resp)
        if fc:
            tool_name = fc.get("name")
            raw_args = fc.get("arguments")
            logger.info("[argue] model requested tool '%s' with args: %s", tool_name, raw_args)

            # run the tool on host
            tool_result = run_tool_and_get_result(tool_name, raw_args)

            # Format tool result as a deterministic text block to send back to model
            try:
                tool_result_text = json.dumps(tool_result) if not isinstance(tool_result, str) else str(tool_result)
            except Exception:
                tool_result_text = str(tool_result)

            # 3) Re-call the model including the tool result as additional user info
            tool_feedback = f"[TOOL_RESPONSE: {tool_name}]\n{tool_result_text}\n[/TOOL_RESPONSE]\nNow, using the above tool output, produce your argument for this round."

            # Combine current instruction and tool feedback into one prompt for the agent
            updated_context_prompt = f"{context_prompt}\n\n{tool_feedback}"

            # **REUSE the corrected builder, passing the new prompt**
            new_contents = self._build_call_contents(last_response, updated_context_prompt)

            resp2 = call_model(self.model, model_name, new_contents)
            final_text = safe_get_text(resp2)
        else:
            final_text = safe_get_text(resp)

        # Save only the last output to keep history small for the next round
        self.history = [self.base_instruction, final_text]
        return final_text


class ContextAwareJudge:
    def __init__(self, thesis_text: str, formatted_references: str, criteria_context: str):
        self.model = CORE_MODEL

        self.instruction = (
            f"You are a neutral, rigorous Judge helping a user improve their thesis. **Be highly detailed in your feedback**.\n"
            f"Thesis: {thesis_text}\n"
            f"Evaluation Criteria: {criteria_context}\n\n"
            f"Context (references):\n{formatted_references}\n\n"
            "**JUDGING CORE RULESET (STRICT ADHERENCE REQUIRED):**\n"
            "**A. Feasibility Focus & Constructive Potential (Primary Rule):**"
            " The primary goal is to validate the thesis idea."
            " Score the 'Research Feasibility' criterion highly for the PRO argument if the research is **academically feasible**"
            " (meaning the research *can* be done, even if the thesis is not overly elaborate on the methodology). Assume user is capable in this regard if in reality, relevant methods exist."
            " Actively look for ways to **fix or re-scope** the thesis to nullify CON's critical claims."
            " Only score CON highly if their criticism is truly fundamental, unfixable, and renders the entire project non-viable."
            " The main point is to help validate *why yes*, not *why not*.\n"
            "**B. Thesis is the Guideline:** The **Thesis is the guideline**."
            " Judge the arguments strictly according to the content and constraints *explicitly stated in the thesis and criteria*,"
            " treating them as the *entire* scope of the research project.\n"
            "SCORING PROTOCOL:\n"
            "- Debate is organized by rounds 1..5. Assign points (1-10) for every speech based on logic, use of references, and alignment with the user's chosen criteria. Round 5 points should reflect the strength of the final synthesis.\n"
            "- Calculate the simple sum of scores for PRO and CON.\n\n"
            "OUTPUT FORMAT (Strict):\n"
            "SUMMARY:\n"
            "- PRO Main Arguments: [Summarize top 3 *distinct* points made throughout the debate, each in new line]\n"
            "- CON Main Concerns: [Summarize top 3 *distinct* concerns made throughout the debate, each in new line]\n\n"
            "SCORES:\n"
            "Round 1: PRO=x, CON=y\n"
            "Round 2: PRO=x, CON=y\n"
            "Round 3: PRO=x, CON=y\n"
            "Round 4: PRO=x, CON=y\n"
            "Round 5: PRO=x, CON=y\n"
            "TOTAL: PRO=x, CON=y\n\n"
            "WINNER: [PRO/CON]\n\n"
            "FEEDBACK FOR USER:\n"
            "Provide 3 actionable steps to improve the thesis based on the CON arguments that won points. **Elaborate on the weaknesses and provide concrete solutions.**"
        )

    def judge(self, transcript: str) -> str:
        """
        Judge should receive the full transcript for final evaluation.
        """
        model_name = getattr(self.model, "model", "gemini-2.5-flash")
        prompt = f"{self.instruction}\n\nTRANSCRIPT OF DEBATE:\n{transcript}"

        max_retries = 3
        for attempt in range(max_retries):
            resp = call_model(self.model, model_name, prompt)
            if resp is not None:
                return safe_get_text(resp)

            print(f"API call failed (returned None), retrying in 2 seconds (Attempt {attempt + 1}/{max_retries})...")
            time.sleep(2)

        raise RuntimeError(f"Failed to get a non-None response from the model after {max_retries} attempts.")


# -----------------------
# Main Process: execute_debate_process
# -----------------------
async def execute_debate_process(
    thesis_text: str,
    references_json: Any,
    runner: Runner, # Explicitly typed Runner now
    user_id: str,
    session_id: str,
    *,
    blocking_mode: str = "to_thread"
):
    """
    Orchestrates:
    1) Criteria dialog (via runner)
    2) 5-round debate between two ContextAwareDebateAgent instances
    3) Final judgment using ContextAwareJudge (gets whole transcript)
    """

    # --- 1. Dialog Phase (Criteria Selection) ---
    async def run_criteria_dialog() -> str:
        initial_msg = (
            f"User Thesis: '{thesis_text}'\n"
            f"References Summary: {references_json}...\n\n"
            "Task: Ask the user to choose exactly 3 criteria from this list:\n"
            "1. Scope and Fit\n2. Academic Relevance/Novelty\n3. Research Feasibility\n"
            "4. Ethical Considerations\n5. Possible Methodology\n6. Professional/Future Relevance\n"
            "7. Personal Interest/Motivation\n\n"
            "After they choose 3, ask if they want to add ONE custom criterion.\n"
            "When finalized, output EXACTLY: 'CRITERIA_FINALIZED: [comma separated list]'"
        )

        current_input = initial_msg

        while True:
            content = gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=current_input)])
            response_stream = runner.run_async(user_id=user_id, session_id=session_id, new_message=content)

            agent_text = ""
            async for event in response_stream:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if getattr(part, "text", None):
                            agent_text += part.text

            if not agent_text:
                agent_text = "..."

            print(f"🤖 **Agent:** {agent_text}")

            if "CRITERIA_FINALIZED:" in agent_text:
                extracted_criteria = agent_text.split("CRITERIA_FINALIZED:", 1)[1].strip()
                return extracted_criteria

            user_response = input("👤 **You:** ").strip()
            current_input = user_response

    print("\n--- 🎯 Criteria Selection Phase ---")
    criteria = await run_criteria_dialog()
    print(f"\n✅ Criteria Selected: {criteria}")

    # --- 2. Debate Phase ---
    # Agent initialization uses the cleaned signature
    con = ContextAwareDebateAgent("Agent CON", "con", thesis_text, references_json, criteria, tools=DEBATE_SEARCH_TOOLS)
    pro = ContextAwareDebateAgent("Agent PRO", "pro", thesis_text, references_json, criteria, tools=DEBATE_SEARCH_TOOLS)

    transcript_lines = [
        f"THESIS: {thesis_text}",
        f"CRITERIA: {criteria}"
    ]

    async def run_agent(agent_obj, prompt_text):
        if blocking_mode == "to_thread":
            return await asyncio.to_thread(agent_obj.argue, prompt_text)
        else:
            return agent_obj.argue(prompt_text)

    last_con_speech = ""
    last_pro_speech = ""

    # ROUND 1
    print("\n--- Round 1: Opening Statements ---")
    r1_prompt = ("ROUND 1: State clearly why this thesis is good/bad based on the criteria."
             "Do not address opponent yet. **You MUST ONLY use the references provided in your context.**"
             "Cite at least one of the provided articles (e.g., 'According to Article \"title\", ...'), "
                 "It should be readable, not in json format."
             "Tool use is strictly forbidden in this round.")
    print("🔵 PRO (R1) Opening:")
    pro_r1 = await run_agent(pro, r1_prompt)
    print(pro_r1)
    transcript_lines.append(f"ROUND 1 PRO: {pro_r1}")
    last_pro_speech = pro_r1

    print("\n🔴 CON (R1) Opening:")
    con_r1 = await run_agent(con, r1_prompt)
    print(con_r1)
    transcript_lines.append(f"ROUND 1 CON: {con_r1}")
    last_con_speech = con_r1

    # ROUND 2
    print("\n--- Round 2: Refute First Arguments (R1 Only) ---")
    print("🔵 PRO (R2) Rebuttal:")
    pro_r2_prompt = f"ROUND 2: **Refute the opponent's R1 claim ONLY.** The opponent's R1 claim was: '{last_con_speech}'"
    pro_r2 = await run_agent(pro, pro_r2_prompt)
    print(pro_r2)
    transcript_lines.append(f"ROUND 2 PRO: {pro_r2}")
    last_pro_speech = pro_r2

    print("\n🔴 CON (R2) Rebuttal:")
    con_r2_prompt = f"ROUND 2: **Refute the opponent's R1 claim ONLY.** The opponent's R1 claim was: '{last_pro_speech}'"
    con_r2 = await run_agent(con, con_r2_prompt)
    print(con_r2)
    transcript_lines.append(f"ROUND 2 CON: {con_r2}")
    last_con_speech = con_r2

    # ROUND 3
    print("\n--- Round 3: Deepening the Argument (Refute R1 & R2) ---")
    print("🔵 PRO (R3) Rebuttal and Strengthen:")
    pro_r3_prompt = f"ROUND 3: **Refute the opponent's R2 claim, and strengthen your case.** You can search for **new research literature** using the academic tool to support your case. The opponent's R2 claim was: '{last_con_speech}'"
    pro_r3 = await run_agent(pro, pro_r3_prompt)
    print(pro_r3)
    transcript_lines.append(f"ROUND 3 PRO: {pro_r3}")
    last_pro_speech = pro_r3

    print("\n🔴 CON (R3) Rebuttal and Strengthen:")
    con_r3_prompt = f"ROUND 3: **Refute the opponent's R2 claim, and strengthen your case.** You can search for **new research literature** using the academic tool to support your case. The opponent's R2 claim was: '{last_pro_speech}'"
    con_r3 = await run_agent(con, con_r3_prompt)
    print(con_r3)
    transcript_lines.append(f"ROUND 3 CON: {con_r3}")
    last_con_speech = con_r3

    # ROUND 4
    print("\n--- Round 4: Complex Rebuttal (Refute R1, R2, R3) ---")
    print("🔵 PRO (R4) Rebuttal and Strengthen:")
    pro_r4_prompt = f"ROUND 4: **Refute the opponent's R3 claim, and strengthen your case.** The opponent's R3 claim was: '{last_con_speech}'"
    pro_r4 = await run_agent(pro, pro_r4_prompt)
    print(pro_r4)
    transcript_lines.append(f"ROUND 4 PRO: {pro_r4}")
    last_pro_speech = pro_r4

    print("\n🔴 CON (R4) Rebuttal and Strengthen:")
    con_r4_prompt = f"ROUND 4: **Refute the opponent's R3 claim, and strengthen your case.** The opponent's R3 claim was: '{last_pro_speech}'"
    con_r4 = await run_agent(con, con_r4_prompt)
    print(con_r4)
    transcript_lines.append(f"ROUND 4 CON: {con_r4}")
    last_con_speech = con_r4

    # ROUND 5
    print("\n--- Round 5: Closing Statements (Final Strengthening) ---")
    # Updated prompt to focus on academic/summary, removing mention of web/statistics
    r5_prompt = "ROUND 5 (FINAL): Ignore the opponent now. Make your final, strongest case for why you are right based on the criteria. You can search for final supporting academic evidence (scholar or pubmed) if needed. Summarize your best points. **Be highly detailed and elaborate**."
    print("🔵 PRO (R5) Final Statement:")
    pro_r5 = await run_agent(pro, r5_prompt)
    print(pro_r5)
    transcript_lines.append(f"ROUND 5 PRO: {pro_r5}")

    print("\n🔴 CON (R5) Final Statement:")
    con_r5 = await run_agent(con, r5_prompt)
    print(con_r5)
    transcript_lines.append(f"ROUND 5 CON: {con_r5}")

    # --- 3. Judging Phase ---
    print("\n⚖️  Judge is deliberating...")
    full_transcript = "\n".join(transcript_lines)

    judge = ContextAwareJudge(thesis_text, references_json, criteria)
    if blocking_mode == "to_thread":
        verdict = await asyncio.to_thread(judge.judge, full_transcript)
    else:
        verdict = judge.judge(full_transcript)

    print("\n🏆 **FINAL VERDICT:**")
    print(verdict)
    return verdict