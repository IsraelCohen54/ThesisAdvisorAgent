# demo_stream_agent.py
import asyncio
import vertexai
from vertexai import agent_engines
from app.config.settings import PROJECT_ID, REGION, AGENT_RESOURCE_ID

async def run_stream(query: str):
    vertexai.init(project=PROJECT_ID, location=REGION)

    # list agents in the region and pick the one matching our resource id (fallback to first)
    agents = list(agent_engines.list())
    if not agents:
        print("No agents found in this region.")
        return

    # try to find the exact deployed agent object
    remote_agent = None
    for a in agents:
        if a.resource_name == AGENT_RESOURCE_ID or AGENT_RESOURCE_ID.endswith(a.resource_name.split("/")[-1]):
            remote_agent = a
            break
    if remote_agent is None:
        remote_agent = agents[0]
        print("Warning: exact resource not found; using first agent:", remote_agent.resource_name)

    print("Using agent:", remote_agent.resource_name)
    print("Streaming response (press Ctrl-C to stop):\n" + "-"*60)

    # stream the response (async iterator)
    async for item in remote_agent.async_stream_query(message=query, user_id="demo_user_1"):
        # item is usually a dict-like object with 'content' -> 'parts' list
        content = getattr(item, "content", None) or item.get("content", {}) if isinstance(item, dict) else {}
        parts = content.get("parts", []) if isinstance(content, dict) else []
        for p in parts:
            # different part shapes: {'text': '...'} or {'function_call': {...}} or {'function_response': {...}}
            if isinstance(p, dict):
                if "text" in p:
                    print(p["text"], end="", flush=True)
                elif "function_call" in p:
                    print("\n[Function call]", p["function_call"])
                elif "function_response" in p:
                    print("\n[Function response]", p["function_response"])
        # optionally print metadata
    print("\n" + "-"*60)
    print("✅ Finished streaming.")

def main():
    query = "Find medical research about the impact of lack of sleep on student grades."
    try:
        asyncio.run(run_stream(query))
    except KeyboardInterrupt:
        print("\nInterrupted.")

if __name__ == "__main__":
    main()
