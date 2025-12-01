import asyncio
import json
import ast
from typing import Any

from google.adk.models.google_llm import Gemini
from google.genai import types as gen_types
from google.genai import types

# Configure Retry
retry_config = types.HttpRetryOptions(
    attempts=5,
    initial_delay=2,
    http_status_codes=[429, 500, 503, 504],
    max_delay=2,
    exp_base=1.5
)


# -----------------------
# Helper: format references
# -----------------------
def format_references_for_context(references_json_str: Any) -> str:
    """
    Parse the incoming references and return a formatted text block.
    """
    try:
        if not references_json_str:
            return "No references provided."

        if isinstance(references_json_str, (list, dict)):
            articles = references_json_str
        else:
            cleaned = str(references_json_str).strip()
            if cleaned.startswith("{") or cleaned.startswith("["):
                try:
                    articles = json.loads(cleaned)
                except json.JSONDecodeError:
                    try:
                        articles = ast.literal_eval(cleaned)
                    except (ValueError, TypeError): # Using specific exceptions
                        articles = cleaned  # Fallback
            else:
                try:
                    articles = ast.literal_eval(cleaned)
                except (ValueError, TypeError): # Using specific exceptions
                    return f"### AVAILABLE REFERENCES / CONTEXT ###\nRaw text:\n{cleaned}\n"

        # Uniform processing
        if isinstance(articles, dict):
            if all(isinstance(k, int) for k in articles.keys()):
                articles = [articles[k] for k in sorted(articles.keys())]
            else:
                articles = [articles]

        # If it failed to become a list by now (e.g. just a string), return text
        if not isinstance(articles, list):
            return f"### AVAILABLE REFERENCES / CONTEXT ###\nRaw text:\n{str(articles)}\n"

        formatted_text = "### AVAILABLE REFERENCES / CONTEXT ###\n\n"

        for idx, art in enumerate(articles, 1):
            if not isinstance(art, dict):
                formatted_text += f"Reference {idx}: {str(art)[:500]}\n\n"
                continue

            title = art.get("title", art.get("name", "Unknown Title"))
            content = art.get("abstract", art.get("summary", art.get("content", "")))
            source = art.get("source", art.get("journal", art.get("publisher", "")))
            authors = art.get("authors", art.get("AU", None))
            link = art.get("link", art.get("url", "No link"))

            formatted_text += f"Article {idx}: '{title}'\n"
            if authors:
                formatted_text += f"   Authors: {authors}\n"
            if source:
                formatted_text += f"   Source:  {source}\n"
            if content:
                snippet = str(content).replace("\n", " ")[:400]
                formatted_text += f"   Details: {snippet}...\n"
            formatted_text += f"   Link:    {link}\n\n"

        return formatted_text

    except Exception as e:
        return f"Error formatting references: {e}\nRaw input (truncated): {str(references_json_str)[:400]}"


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
    ):
        self.name = name
        self.stance = stance.upper()  # "PRO" or "CON"

        # Base Persona instruction (general)
        self.base_instruction = (
            f"You are {name}. Stance: {self.stance}. Your goal is to be highly persuasive.\n"
            f"Thesis Topic: {thesis_text}\n"
            f"Evaluation Criteria chosen by user: {criteria_context}\n\n"
            f"{formatted_references}\n\n"
            "GENERAL GUIDELINES:\n"
            "1. Be persuasive based on the criteria chosen.\n"
            "2. Use specific details from the provided references to support your point.\n"
            "3. Maintain professional, concise, and logical consistency. **Be highly detailed and elaborate**."
        )

        self.model = Gemini(
            model="gemini-2.5-flash",
            retry_options=retry_config,
        )

        # History will now only store the final response of each turn, NOT the accumulating prompt
        # We start with a placeholder for the base instruction
        self.history = [self.base_instruction]

    def argue(self, context_prompt: str) -> str:
        """
        Generates an argument.
        'context_prompt' contains the specific instructions for the current round
        and the opponent's immediate previous point.
        """

        # 1. Build the full, current prompt for this specific call
        # Start with the base instruction to set the scene
        prompt_parts = [self.base_instruction, "\n\n--- PREVIOUS ROUNDS' RESPONSES ---\n"]

        # Add the entire debate history (previous responses only, no prompts)
        # Start from 1 to skip the base instruction, which is already at the start
        for i, h in enumerate(self.history[1:]):
            prompt_parts.append(f"Response R{i+1}: {h}")

        # Add the specific instruction for the current round
        prompt_parts.append(f"\n\n--- CURRENT ROUND INSTRUCTION ---\n{context_prompt}")

        full_prompt = "\n\n".join(prompt_parts)

        # 2. Prepare the content object for the API call
        content_to_send = [gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=full_prompt)])]

        # 3. Call Gemini (Sync)
        model_name = getattr(self.model, "model", "gemini-2.5-flash")
        resp = self.model.api_client.models.generate_content(model=model_name, contents=content_to_send)

        text = getattr(resp, "text", str(resp))

        # 4. Update internal history (store the model's output)
        self.history.append(text)
        return text


class ContextAwareJudge:
    def __init__(self, thesis_text: str, formatted_references: str, criteria_context: str):
        self.model = Gemini(model="gemini-2.5-flash", retry_options=retry_config)

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
        model_name = getattr(self.model, "model", "gemini-2.5-flash")
        prompt = f"{self.instruction}\n\nTRANSCRIPT OF DEBATE:\n{transcript}"
        resp = self.model.api_client.models.generate_content(model=model_name, contents=prompt)
        return getattr(resp, "text", str(resp))


# -----------------------
# Main Process
# -----------------------
async def execute_debate_process(
        thesis_text: str,
        references_json: Any,
        runner,
        user_id: str,
        session_id: str,
        *,
        blocking_mode: str = "to_thread"
):
    formatted_refs = format_references_for_context(references_json)

    # --- 1. Dialog Phase (Criteria Selection) ---
    async def run_criteria_dialog() -> str:
        initial_msg = (
            f"User Thesis: '{thesis_text}'\n"
            f"References Summary: {formatted_refs[:300]}...\n\n"
            "Task: Ask the user to choose exactly 3 criteria from this list:\n"
            "1. Scope and Fit\n2. Academic Relevance/Novelty\n3. Research Feasibility\n"
            "4. Ethical Considerations\n5. Possible Methodology\n6. Professional/Future Relevance\n"
            "7. Personal Interest/Motivation\n\n"
            "After they choose 3, ask if they want to add ONE custom criterion.\n"
            "When finalized, output EXACTLY: 'CRITERIA_FINALIZED: [comma separated list]'"
        )

        # Start the conversation
        current_input = initial_msg

        # We need to loop until the agent finalizes
        while True:
            # Send message to agent
            content = gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=current_input)])
            response_stream = runner.run_async(user_id=user_id, session_id=session_id, new_message=content)

            agent_text = ""
            async for event in response_stream:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            agent_text += part.text

            if not agent_text:
                agent_text = "..."

            print(f"🤖 **Agent:** {agent_text}")

            # Check for termination signal
            if "CRITERIA_FINALIZED:" in agent_text:
                # Extract the criteria string after the colon
                extracted_criteria = agent_text.split("CRITERIA_FINALIZED:", 1)[1].strip()
                return extracted_criteria

            user_response = input("👤 **You:** ").strip()
            current_input = user_response

    print("\n--- 🎯 Criteria Selection Phase ---")
    criteria = await run_criteria_dialog()
    print(f"\n✅ Criteria Selected: {criteria}")

    # --- 2. Debate Phase ---
    con = ContextAwareDebateAgent("Agent CON", "con", thesis_text, formatted_refs, criteria)
    pro = ContextAwareDebateAgent("Agent PRO", "pro", thesis_text, formatted_refs, criteria)

    transcript_lines = [
        f"THESIS: {thesis_text}",
        f"CRITERIA: {criteria}"
    ]

    # Helper to run blocking call
    async def run_agent(agent_obj, prompt_text):
        if blocking_mode == "to_thread":
            return await asyncio.to_thread(agent_obj.argue, prompt_text)
        else:
            return agent_obj.argue(prompt_text)

    # Store previous responses for the next round's rebuttal instruction
    last_con_speech = ""
    last_pro_speech = ""

    # --- ROUND 1: Opening Statement (Why good/not good) ---
    print("\n--- Round 1: Opening Statements ---")
    r1_prompt = ("ROUND 1: State clearly why this thesis is good/bad based on the criteria."
                 "Do not address opponent yet. **Find and cite SPECIFIC, current statistics"
                 "(e.g., desalination capacity in cubic meters, cost per cubic meter)"
                 "using the search tool** to support your claim. Be highly detailed and elaborate."
                 "if it is feasible, if using method that isn't written explicitly in thesis, add them as support."
                 "")\
    # PRO Opens
    print("🔵 PRO (R1) Opening:")
    pro_r1 = await run_agent(pro, r1_prompt)
    print(pro_r1)
    transcript_lines.append(f"ROUND 1 PRO: {pro_r1}")
    last_pro_speech = pro_r1

    # CON Responds
    print("\n🔴 CON (R1) Opening:")
    con_r1 = await run_agent(con, r1_prompt)
    print(con_r1)
    transcript_lines.append(f"ROUND 1 CON: {con_r1}")
    last_con_speech = con_r1

    # --- ROUND 2: Refute Round 1 ONLY ---
    print("\n--- Round 2: Refute First Arguments (R1 Only) ---")

    # PRO Refutes CON's R1
    print("🔵 PRO (R2) Rebuttal:")
    # last_con_speech is USED HERE (as the opponent's R1 claim)
    pro_r2_prompt = f"ROUND 2: **Refute the opponent's R1 claim ONLY.** The opponent's R1 claim was: '{last_con_speech}'"
    pro_r2 = await run_agent(pro, pro_r2_prompt)
    print(pro_r2)
    transcript_lines.append(f"ROUND 2 PRO: {pro_r2}")
    last_pro_speech = pro_r2

    # CON Refutes PRO's R1
    print("\n🔴 CON (R2) Rebuttal:")
    # last_pro_speech is USED HERE (as the opponent's R1 claim)
    con_r2_prompt = f"ROUND 2: **Refute the opponent's R1 claim ONLY.** The opponent's R1 claim was: '{last_pro_speech}'"
    con_r2 = await run_agent(con, con_r2_prompt)
    print(con_r2)
    transcript_lines.append(f"ROUND 2 CON: {con_r2}")
    last_con_speech = con_r2

    # --- ROUND 3: Refute Round 1 + 2 ---
    print("\n--- Round 3: Deepening the Argument (Refute R1 & R2) ---")

    # PRO Refutes CON's R2 & strengthens
    print("🔵 PRO (R3) Rebuttal and Strengthen:")
    # last_con_speech is USED HERE (as the opponent's R2 claim)
    pro_r3_prompt = f"ROUND 3: **Refute the opponent's R2 claim, and strengthen your case.** The opponent's R2 claim was: '{last_con_speech}'"
    pro_r3 = await run_agent(pro, pro_r3_prompt)
    print(pro_r3)
    transcript_lines.append(f"ROUND 3 PRO: {pro_r3}")
    last_pro_speech = pro_r3

    # CON Refutes PRO's R2 & strengthens
    print("\n🔴 CON (R3) Rebuttal and Strengthen:")
    # last_pro_speech is USED HERE (as the opponent's R2 claim)
    con_r3_prompt = f"ROUND 3: **Refute the opponent's R2 claim, and strengthen your case.** The opponent's R2 claim was: '{last_pro_speech}'"
    con_r3 = await run_agent(con, con_r3_prompt)
    print(con_r3)
    transcript_lines.append(f"ROUND 3 CON: {con_r3}")
    last_con_speech = con_r3

    # --- ROUND 4: Complex Synthesis (Refute R1, R2, R3) ---
    print("\n--- Round 4: Complex Rebuttal (Refute R1, R2, R3) ---")

    # PRO Refutes CON's R3 & strengthens
    print("🔵 PRO (R4) Rebuttal and Strengthen:")
    # last_con_speech is USED HERE (as the opponent's R3 claim)
    pro_r4_prompt = f"ROUND 4: **Refute the opponent's R3 claim, and strengthen your case.** The opponent's R3 claim was: '{last_con_speech}'"
    pro_r4 = await run_agent(pro, pro_r4_prompt)
    print(pro_r4)
    transcript_lines.append(f"ROUND 4 PRO: {pro_r4}")
    last_pro_speech = pro_r4

    # CON Refutes PRO's R3 & strengthens
    print("\n🔴 CON (R4) Rebuttal and Strengthen:")
    # last_pro_speech is USED HERE (as the opponent's R3 claim)
    con_r4_prompt = f"ROUND 4: **Refute the opponent's R3 claim, and strengthen your case.** The opponent's R3 claim was: '{last_pro_speech}'"
    con_r4 = await run_agent(con, con_r4_prompt)
    print(con_r4)
    transcript_lines.append(f"ROUND 4 CON: {con_r4}")
    last_con_speech = con_r4

    # --- ROUND 5: Final Strengthening (No regard for opponent) ---
    print("\n--- Round 5: Closing Statements (Final Strengthening) ---")
    r5_prompt = "ROUND 5 (FINAL): Ignore the opponent now. Make your final, strongest case for why you are right based on the criteria. Summarize your best points. **Be highly detailed and elaborate**."

    # PRO Final Statement
    print("🔵 PRO (R5) Final Statement:")
    pro_r5 = await run_agent(pro, r5_prompt)
    print(pro_r5)
    transcript_lines.append(f"ROUND 5 PRO: {pro_r5}")

    # CON Final Statement
    print("\n🔴 CON (R5) Final Statement:")
    con_r5 = await run_agent(con, r5_prompt)
    print(con_r5)
    transcript_lines.append(f"ROUND 5 CON: {con_r5}")

    # --- 3. Judging Phase ---
    print("\n⚖️  Judge is deliberating...")
    full_transcript = "\n".join(transcript_lines)

    judge = ContextAwareJudge(thesis_text, formatted_refs, criteria)
    if blocking_mode == "to_thread":
        verdict = await asyncio.to_thread(judge.judge, full_transcript)
    else:
        verdict = judge.judge(full_transcript)

    print("\n🏆 **FINAL VERDICT:**")
    print(verdict)
    return verdict