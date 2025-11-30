# anylize_and_recommend.py
import asyncio
import json
import ast
from typing import Optional, Any

from google.adk.models.google_llm import Gemini
from google.genai import types as gen_types


# -----------------------
# Helper: format references
# -----------------------
def format_references_for_context(references_json_str: Any) -> str:
    """
    Parse the incoming references (string/json/list/dict) and return a nicely
    formatted text block suitable for LLM context.

    Accepts:
      - JSON string (double quotes)
      - Python dict/list string (single quotes) -> ast.literal_eval
      - Already-parsed list/dict
      - None/empty -> returns a short message
    """
    try:
        if not references_json_str:
            return "No references provided."

        # If already a dict/list, use it directly
        if isinstance(references_json_str, (list, dict)):
            articles = references_json_str
        else:
            # It's probably a string, so try JSON first, then literal_eval
            cleaned = str(references_json_str).strip()
            if cleaned.startswith("{") or cleaned.startswith("["):
                try:
                    articles = json.loads(cleaned)
                except json.JSONDecodeError:
                    # Try Python literal_eval (handles single quotes)
                    articles = ast.literal_eval(cleaned)
            else:
                # Not JSON-like: try literal_eval, else treat as plain text
                try:
                    articles = ast.literal_eval(cleaned)
                except Exception:
                    # fallback: return the raw text as one "reference"
                    return f"### AVAILABLE REFERENCES / CONTEXT ###\nRaw text:\n{cleaned}\n"

        # At this point, expect articles to be a list of dicts (or a dict)
        if isinstance(articles, dict):
            # Possibly a single article or results wrapper
            # If it's a mapping with numeric keys, convert to list
            if all(isinstance(k, int) for k in articles.keys()):
                articles = [articles[k] for k in sorted(articles.keys())]
            else:
                # Wrap single dict into a list for uniform processing
                articles = [articles]

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
        # Safe fallback: show a short message
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
            api_key: Optional[str] = None,
    ):
        self.name = name
        self.stance = stance  # "pro" or "con"
        self.instruction = (
            f"You are {name}. Stance: {stance.upper()}.\n"
            f"Thesis Topic: {thesis_text}\n"
            f"Evaluation Criteria: {criteria_context}\n\n"
            f"{formatted_references}\n\n"
            "INSTRUCTIONS:\n"
            "1. You should be as persuade as you can According to criteria chosen (keep using only the truth).\n"
            "2. If relevant, use the specific details from the references above (articles) to support your point.\n"
            "3. Refute the opponent's last argument based on any available source and logic (use only logical truths)."
            "4. \nKeep it professional, concise, and cite article data if relevant."
        )

        self.model = Gemini(model="gemini-2.5-flash-lite", api_key=api_key)
        self.history = [gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=self.instruction)])]

    def argue(self, opponent_argument: Optional[str] = None) -> str:
        """
        Blocking (sync) call that uses the lower-level synchronous client API.
        We build a prompt from the stored history and call the API synchronously.
        This will be safe to run inside asyncio.to_thread(...) as before.
        """
        # Build a textual prompt from history (fallback if ADK content shapes vary)
        parts = []
        for content in self.history:
            if getattr(content, "parts", None):
                for p in content.parts:
                    if getattr(p, "text", None):
                        parts.append(p.text)
        # Append the opponent argument / new prompt
        if opponent_argument:
            parts.append(f"Opponent said: {opponent_argument}\nYour rebuttal/point:")
        else:
            parts.append("Present your opening argument based on the references.")

        prompt = "\n\n".join(parts)

        # Use the lower-level sync API on the generated api_client
        # Note: pass the model name used when creating the Gemini wrapper.
        model_name = getattr(self.model, "model", "gemini-2.5-flash-lite")
        resp = self.model.api_client.models.generate_content(model=model_name, contents=prompt)

        text = getattr(resp, "text", str(resp))

        # Append to history to keep context for future turns
        self.history.append(gen_types.Content(role="model", parts=[gen_types.Part.from_text(text=text)]))
        return text


class ContextAwareJudge:
    def __init__(self, thesis_text: str, formatted_references: str, criteria_context: str,
                 api_key: Optional[str] = None):
        self.model = Gemini(model="gemini-2.5-flash", api_key=api_key)
        self.instruction = (f""""You are a neutral and rigorous Judge.
You must decide whether the PRO or CON debater has presented the stronger case for this thesis idea.

Thesis: {thesis_text}
Evaluation Criteria: {criteria_context}

Context (reference summaries and evidence):
{formatted_references}

JUDGING RULES:
1. Judge only the strength, clarity, and logical quality of the arguments presented by PRO and CON based on {criteria_context}.
2. Do NOT assume that negative arguments are stronger by default. Treat PRO and CON with equal burden of proof.
3. When both sides make reasonable equal points, **lean toward PRO** as such identical thesis wasn't found at prior step,
   unless the thesis has a clear, concrete or unfixable flaw.
4. References may support arguments but are NOT the deciding factor; the deciding factor is logical reasoning and argument strength.

DECISION FORMAT:
- First: Announce the winner: PRO or CON.
- Second: Provide a short justification grounded strictly in logic according to the stated criteria.

GUIDANCE WHEN CON WINS:
If you choose CON:
- You MUST also help the user.
- Provide constructive steps for how the thesis could be improved, reframed, narrowed, or applied in a context where it *would* meet at least some criteria.
- Emphasize salvageable elements rather than total rejection.

GUIDANCE WHEN PRO WINS:
If you choose PRO:
- Briefly explain what aspects make the thesis promising and feasible, and what strengths stood out.
""")

    def judge(self, transcript: str) -> str:
        """
        Synchronous judge call using the lower-level sync API.
        Safe to run in asyncio.to_thread(...) or directly in blocking mode.
        """
        model_name = getattr(self.model, "model", "gemini-2.5-flash")
        prompt = f"{self.instruction}\n\nTRANSCRIPT:\n{transcript}"
        resp = self.model.api_client.models.generate_content(model=model_name, contents=prompt)
        return getattr(resp, "text", str(resp))


# -----------------------
# Main: execute_debate_process (async)
# -----------------------
async def execute_debate_process(
        thesis_text: str,
        references_json: Any,
        runner,
        user_id: str,
        session_id: str,
        api_key: Optional[str] = None,
        *,
        blocking_mode: str = "to_thread"
):
    """
    Run: dialog (criteria selection using provided runner & session) -> debate -> judge.
    The caller awaits this function and receives the final verdict.
    """
    formatted_refs = format_references_for_context(references_json)

    # -----------------
    #   Dialog phase
    # -----------------
    async def run_criteria_dialog_with_runner() -> str:
        talk_instruction = f"""
You are 'DialogAgent'.
The user is working on this thesis: "{thesis_text}".

Context/References available:
{formatted_refs}

Your Goal: Ask the user to choose the 3 most important evaluation criteria from:
    1. "Scope and Fit"
    2. "Academic Relevance and Novelty"
    3. "Research Feasibility"
    4. "Ethical Considerations"
    5. "Possible Methodology"
    6. "Professional and Future Relevance"
    7. "Personal Interest and Motivation" 

Protocol:
1. Ask the user to select 3.
2. Only after user answer, ask if they want to add ONE custom criterion.
3. When finalized, output ONLY: "CRITERIA_FINALIZED: [The list]"
"""
        user_input = talk_instruction
        final_criteria = ""

        while True:
            response_stream = runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=gen_types.Content(role="user", parts=[gen_types.Part.from_text(text=user_input)])
            )

            agent_response = None
            async for event in response_stream:
                content = getattr(event, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        if getattr(part, "text", None):
                            agent_response = part.text

            if not agent_response:
                agent_response = "Error: No response from agent."

            print(f"\n🤖 **DialogAgent:** {agent_response}")

            if "CRITERIA_FINALIZED" in agent_response:
                final_criteria = agent_response.replace("CRITERIA_FINALIZED:", "").strip()
                break

            user_input = input("\n👤 **You:** ").strip()

        return final_criteria

    # run dialog and get criteria
    criteria = await run_criteria_dialog_with_runner()

    # -----------------------
    # Debate phase
    # -----------------------
    pro = ContextAwareDebateAgent("Agent PRO", "pro", thesis_text, formatted_refs, criteria, api_key=api_key)
    con = ContextAwareDebateAgent("Agent CON", "con", thesis_text, formatted_refs, criteria, api_key=api_key)

    transcript = ""
    last_arg = None

    for i in range(5):
        print(f"\n--- Round {i + 1} ---")

        if blocking_mode == "to_thread":
            last_arg = await asyncio.to_thread(pro.argue, last_arg)
        else:
            last_arg = pro.argue(last_arg)

        print(f"🔵 PRO: {last_arg}")
        transcript += f"PRO: {last_arg}\n"
        # short non-blocking pause so stdout prints cleanly
        await asyncio.sleep(0.1)

        if blocking_mode == "to_thread":
            last_arg = await asyncio.to_thread(con.argue, last_arg)
        else:
            last_arg = con.argue(last_arg)

        print(f"🔴 CON: {last_arg}")
        transcript += f"CON: {last_arg}\n"
        await asyncio.sleep(0.1)

    # -----------------------
    # Judging (blocking)
    # -----------------------
    judge = ContextAwareJudge(thesis_text, formatted_refs, criteria, api_key=api_key)
    if blocking_mode == "to_thread":
        verdict = await asyncio.to_thread(judge.judge, transcript)
    else:
        verdict = judge.judge(transcript)

    print("\n🏆 **VERDICT:**")
    print(verdict)
    return verdict
