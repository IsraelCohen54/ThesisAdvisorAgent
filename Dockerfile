# Example base image (yours may be different, but keep the base image line)
FROM python:3.11-slim

# --- ADD THIS LINE ---
# This ensures the SERPAPI_API_KEY environment variable is set permanently in the container.
ENV SERPAPI_API_KEY=""
# ---------------------

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set entrypoint (this will be handled by the ADK framework, but often required)
# ENTRYPOINT ["/usr/local/bin/python", "-m", "google.adk.runtime"]
