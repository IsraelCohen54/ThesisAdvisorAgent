🎓 Thesis Advisor AI Agent
An intelligent, multi-agent system designed to help researchers validate, refine, and stress-test their thesis ideas using real academic data and adversarial debate.

📖 Overview
The Thesis Advisor AI is not just a chatbot; it is a structured workflow designed to mimic a rigorous academic consultation. It leverages Google Cloud Vertex AI and the Agent Development Kit (ADK) to perform a two-phase analysis:

Discovery Phase: A specialized "Talk Agent" searches for real academic literature (PubMed, Google Scholar) relevant to your thesis.

Debate Phase: Two context-aware agents (PRO and CON) engage in a multi-round debate to analyze the feasibility, novelty, and ethics of your idea, supervised by a "Judge" agent that provides a final verdict and scoring.

🚀 Key Features
Hybrid Architecture: Combines deployed Vertex AI agents for tool execution with local, asynchronous coordination for complex debate logic.

Real Academic Grounding: Uses custom FunctionTools to query Google Scholar and PubMed, ensuring advice is based on existing literature, not hallucinations.

Multi-Agent Debate System:

Agent PRO: Argues for the feasibility and impact of the thesis.

Agent CON: Critiques methodology, scope, and ethical implications.

Context-Aware: Agents remember the opponent's previous points but adhere to strict round-by-round rules (e.g., Opening, Rebuttal, Closing).

Automated Judgment: A neutral Judge agent reviews the entire debate transcript to assign scores and suggest actionable improvements.

Structured Output: Parses complex tool outputs into clean, readable citations and snippets.

🛠️ Architecture
The project is structured into Core logic and Infrastructure:

Plaintext

ThesisAdvisorAgent/<br>
├── app//<br>
│   ├── config//<br>
│   │   └── settings.py           # Configuration (Model versions, Logging)/<br>
│   ├── core//<br>
│   │   ├── agents.py             # Definition of Tool-Use Agents (Scholar/PubMed)/<br>
│   │   └── anylize_and_recommend.py # The Debate & Judge Logic (PRO/CON/Judge)/<br>
│   └── infrastructure//<br>
│       └── tools.py              # Implementation of GoogleScholar and PubMed clients/<br>
├── thesis_advisor_client.py      # Main entry point (CLI application)/<br>
├── requirements.txt              # Dependencies/<br>
└── README.md/<br>
📦 Prerequisites
Before running the agent, ensure you have:

Python 3.10+ installed.

A Google Cloud Project with Vertex AI API enabled.

Google Cloud CLI installed and authenticated (gcloud auth application-default login).

🔧 Installation
Clone the Repository

Bash

git clone https://github.com/yourusername/thesis-advisor-agent.git
cd thesis-advisor-agent
Create a Virtual Environment

Bash

python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
Install Dependencies

Bash

pip install -r requirements.txt
Environment Setup The application relies on Google Cloud environment variables. You can set them in your terminal or ensure thesis_advisor_client.py has the correct defaults:

Bash

export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
🏃 Usage
Run the main client application:

Bash

python thesis_advisor_client.py
The Workflow:
Input: Enter your thesis idea (e.g., "The impact of microplastics on gut microbiome diversity in coastal regions").

Search: The agent will use PubMed (for medical topics) or Google Scholar (for general topics) to find 5 relevant papers.

Review: You will see a formatted list of articles with snippets and links.

Action: You can choose to:

[Q]uit: Exit.

[R]efine: Rewrite your thesis and search again.

[C]ontinue: Launch the Debate.

The Debate Process (If you choose 'Continue'):
Criteria Selection: The agent asks you to select evaluation criteria (e.g., Feasibility, Novelty, Ethics).

Rounds 1-5: PRO and CON agents argue. They will utilize the academic references found in step 2 to support their claims.

Verdict: The Judge provides a final summary, scores for each side, and concrete advice on how to improve your thesis.

🛡️ Security Note
This repository contains agent logic (agents.py) but excludes specific deployment configurations (.agent_engine_config.json) to protect project credentials.

When deploying or running locally, ensure your GOOGLE_CLOUD_PROJECT environment variables are set correctly. Never commit your service account keys or API keys to GitHub.

🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the project.

Create your feature branch (git checkout -b feature/AmazingFeature).

Commit your changes.

Push to the branch.

Open a Pull Request.

📄 License
Distributed under the MIT License. See LICENSE for more information.
