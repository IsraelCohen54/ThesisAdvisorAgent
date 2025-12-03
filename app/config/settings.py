# app/config/settings.py

import logging
import uuid

from google.genai import types
from google.adk.models.google_llm import Gemini


# 1. Logging Configuration
APP_NAME = "ThesisAdvisor"
logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.INFO)

# --- ADK/Gemini Configuration ---
# 2. API Retry Configuration
RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,
    initial_delay=2,
    http_status_codes=[429, 500, 503, 504],
    max_delay=2,
    exp_base=1.5
)

# 3. Model Definition
# Core model object using the retry config
CORE_MODEL = Gemini(model="gemini-2.5-flash", retry_options=RETRY_CONFIG)

# 4. --- Vertex AI Deployment Constants ---
REGION =
PROJECT_ID =
AGENT_RESOURCE_ID = f"projects/682003720850/locations/{REGION}/reasoningEngines/5242590188491767808"

# 5. For PubMed Search tool - "Entrez" configuration
EMAIL = "hartk111@gmail.com"

# 6. Minimum length for thesis input:
THESIS_MINIMUM_LENGHT = 10

# Probably on real deploy you would have to change it
USER_ID = "user_1"
SESSION_ID = f"session_{uuid.uuid4().hex[:6]}"