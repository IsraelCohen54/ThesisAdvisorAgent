# thesis_advisor_deploy/agent.py
import os
import vertexai
# from app.core.agents import get_dialog_agent1 
# from thesis_advisor_deploy_tmp20251201_192512.app.core.agents import get_dialog_agent1
from .app.core.agents import get_dialog_agent1

# Initialize Vertex AI 
vertexai.init(
    project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
)

# Call the correctly named function
root_agent = get_dialog_agent1()