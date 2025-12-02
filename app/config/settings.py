# app/config/settings.py

import logging
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
REGION = "us-central1"
PROJECT_ID =
AGENT_RESOURCE_ID =

# 5. For PubMed Search tool - "Entrez" configuration
EMAIL = "hartk111@gmail.com"