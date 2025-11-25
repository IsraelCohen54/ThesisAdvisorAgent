import os
import sys


def verify_step_1():
    print("--- 🛠️ Agile Step 1: Foundation Check ---")

    # 1. Check API Key (without dotenv)
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("❌ Error: API Key not found. Did you set it in PyCharm 'Edit Configurations'?")
        return
    else:
        print("✅ API Key detected in environment.")

    # 2. Verify Imports
    print("⏳ Testing ADK Imports...")
    try:
        # These are the imports you requested
        from google.adk.models.google_llm import Gemini
        from google.adk.tools.agent_tool import AgentTool
        from google.adk.tools.google_search_tool import google_search

        # NOTE: 'LlmAgent' is often just 'Agent' in recent ADK versions.
        # We test both to be safe.
        try:
            from google.adk.agents import LlmAgent
            print("✅ Import 'LlmAgent' successful.")
        except ImportError:
            from google.adk.agents import Agent
            print("⚠️ 'LlmAgent' not found, but 'Agent' is available (Using standard ADK class).")

        print("✅ All critical libraries installed.")

    except ImportError as e:
        print(f"❌ Import Failed: {e}")
        print("Action: Run 'pip install google-adk'")
        return

    # 3. Quick Connection Test
    try:
        print("⏳ Pinging Gemini Model...")
        model = Gemini(model="gemini-2.0-flash-exp")
        # We don't run a prompt yet, just checking initialization
        print("✅ Model initialized successfully.")

        # 2. Define the Prompt (Your question)
        user_prompt = "Explain what a binary search tree is in simple terms, using an analogy."
        print(f"\n❓ User Prompt: **{user_prompt}**")

        # 3. Use the model's generate_content method for a direct response
        try:
            # generate_content() is the method for simple, single-turn requests.
            response = model(contents=user_prompt)

            # 4. Extract and Display the Answer
            print("\n💡 Gemini Response:")
            print(response.text)

        except Exception as e:
            print(f"❌ Error running generate_content: {e}")

    except Exception as e:
        print(f"❌ Model Init Error: {e}")


if __name__ == "__main__":
    verify_step_1()

