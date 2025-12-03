# 🎓 Thesis Advisor AI Agent


*Our Explanation Video*
video https://www.youtube.com/watch?v=2haQXxLy46E&embeds_referring_euri=https%3A%2F%2Fwww.kaggle.com%2F&embeds_referring_origin=https%3A%2F%2Fwww.kaggle.com&source_ve_path=OTY3MTQ

# Thesis Advisor 
An interactive multi-agent assistant that helps students choose, refine, and evaluate thesis ideas by searching most similar literature, and staging a structured pro/con debate over the thesis idea using the retrieved literature and user's evaluation criteria.

## Problem statement - What we’re solving and why does it matter
Choosing a thesis topic is a multi-year commitment, and therefore warrants careful, deliberate consideration.

However, it can be challenging to determine whether a potential topic is truly innovative, feasible, and ultimately fulfilling.
Difficulties in selecting a thesis topic may arise from challenges, such as:
1. Efficiently locating and synthesizing relevant scholarly literature
2. The need to evaluate ideas based on subjective criteria, rather than standardized guidelines
3. Anticipating methodological risks and constraints
4. Obtaining balanced, evidence-based perspectives on the proposed direction

## Why agents? Why a multi-agent design fits
We used an agent-based architecture because complex decision-making tasks (such as evaluating a thesis topic), are inherently multi-dimensional and cannot be reliably addressed through a single, monolithic LLM response.

Agents allow us to decompose such a task into smaller, well-defined components, with each agent specializing in a distinct cognitive function.

This structured division of labor produces results that are more accurate, transparent, and aligned with the user’s needs.

The multi-agent architecture offers several significant advantages, enabling deep collaboration between the user and specialized AI roles:

1. Focused and intelligent information retrieval.
The agent-based approach goes beyond general search, which often overwhelms users with countless articles.
Retrieval agents actively explore, filter, and prioritize the most relevant and conceptually similar sources.

2. Personalized evaluation based on subjective criteria.
A dedicated dialogue agent helps the user define personalized evaluation criteria.
This ensures that the final recommendation reflects the user’s subjective priorities rather than generic, one-size-fits-all guidelines.

3. Structured reasoning through debate.
The debate framework introduces explicit reasoning steps and counterarguments.
Pro and con agents critically examine the thesis topic, retrieved literature, and user criteria from opposing perspectives.
This process yields a more precise, balanced, and rigorous outcome, substantially more refined than a simple single-shot LLM answer.

4. Judgment. A Judgment Agent evaluates the entire debate transcript.
It scores the arguments based on how well they utilized the provided literature and met the user-defined criteria while staying true to their role.
This structured judgment generates a much stronger, more holistic synthesis and a refined final answer.
This step provides the accountability and clarity needed for the user to understand why the topic is recommended or rejected.

5. Modularity and transparency.
Specialized agent roles mirror human committee workflows.
This modularity enables improving one component without affecting others, while giving users insight into the intermediate reasoning steps (debate arguments, justification scores, etc.).

In summary, the agent architecture transforms a complex, subjective, knowledge-heavy decision task into a transparent, analyzable, and user-aligned process.
This makes agents not only appropriate but essential for achieving reliable, high-quality results in this context.

## Overall architecture
<img width="2219" height="1137" alt="flow all" src="https://github.com/user-attachments/assets/b8cf148f-08ae-4ef7-8245-6c8c8e57a4b8" />

Core to the Thesis Advisor is a robust multi-agent ecosystem designed to simulate a rigorous academic review committee. It moves beyond a monolithic LLM interaction by orchestrating a team of specialized agents, each contributing to a specific stage of the research validation process. This modular approach, facilitated by Google’s Agent Development Kit (ADK), allows for a sophisticated workflow that deliberately separates information retrieval from critical analysis.

1. The Search Dispatcher: DialogAgent The entry point of the intelligent workflow is the Root Dialog Agent. Acting as a strict router, this agent analyzes the semantic context of the user’s thesis topic to determine the appropriate academic domain. It utilizes specialized FunctionTools to connect the system to external knowledge bases.
No hallucinating sources! It executes a deterministic logic to dispatch the query to either the PubMed Connector (for biomedical topics) or the Google Scholar Connector (for general academic research), ensuring the downstream debate is grounded in real-world literature. This search dispatcher service has been deployed to the Google Cloud.

2. The Adversarial Engine: ContextAwareDebateAgent (PRO & CON) Once the literature is retrieved, the system initiates the debate subsystem, a structured adversarial loop involving two distinct agent personas:

3. Agent PRO & Agent CON: These agents operate in parallel over five refinement rounds, while remembering only the immediate preceding round, a design choice that promotes fairness by preventing an accumulated "last word" advantage. They are designed to simulate academic researchers, evolving their arguments from initial claims (constrained to user-selected criteria) to deep, evidence-based rebuttals.

4. Dynamic Tool Use: Unlike static chatbots, these agents have permission to access search tools during specific "Deepening" and "Closing" rounds to find new evidence dynamically as the argument evolves.

5. The Adjudicator: ContextAwareJudge, the final component is the impartial Judge. This agent does not participate in the argument but acts as an objective observer. It consumes the aggregated transcript of the debate to perform a high-level synthesis. Its instruction set directs it to score the quality of the argumentation, declare a winner based on evidence usage, and translate the complex exchange into actionable steps for the user.

6. Orchestration & State: The entire lifecycle is managed by a Runner utilizing an InMemorySessionService, which maintains the ephemeral state of the conversation and context across the search, criteria selection, and debate phases, ensuring a seamless user journey from raw idea to validated verdict.

## Demo: User Journey & Execution Flow
(A 1-minute demonstration video showcasing the User Journey [https://youtu.be/mErxCkN1oow])
1. Initialization & Semantic Routing The user launches the authenticated CLI and inputs a raw thesis idea. The Root Agent analyzes the input semantic context and intelligently dispatches the query to the single most appropriate tool connector:
* Biomedical topics => PubMedTool (via NCBI Entrez).
* General topics => GoogleScholarTool (via SerpApi).
2. Retrieval & User Review The selected tool executes a live search, returning structured results. The CLI displays the top references (Title, Author, Snippet). The user enters a decision loop: [Q]uit, [R]efine (restart with new input), or [C]ontinue to the debate.
3. Criteria Selection Upon continuing, a dedicated Dialog Agent guides the user to select exactly three specific evaluation criteria (e.g., Feasibility, Novelty, Ethics) to constrain the upcoming debate.
4. The Adversarial Debate Loop (5 Rounds) Two agents (PRO and CON) engage in a structured debate.
To ensure efficiency and robustness and fairness, agents utilize Context Compaction, retaining only the most recent round's memory. (*Attached image that show the debate rules and flow*).
<img width="1880" height="1037" alt="inbox_29340440_e631a897dd32437a78231e102e373e82_debaters_flow" src="https://github.com/user-attachments/assets/b64d44f9-91fc-41fb-9685-fcc3b7ad9296" />

## The build - tools & technologies
### Core Technology Stack:
1. Language & Concurrency: Python 3.11 with heavy use of asyncio for performance and handling concurrent operations.
2. LLM Framework: Built on the Google ADK and GenAI SDK.
3. Model: Powered by Gemini-2.5-flash (fast, complex reasoning, tool use).
4. Deployment: Managed using Vertex AI for a scalable runtime environment.

### Multi-Agent Architecture:
#### The system uses a Debate + Adjudication pattern involving three core agent roles:
1. Dialog Agent: Manages the initial conversation flow, working sequentially.
2. Debating Agents (Pro/Con): Two separate agents run in parallel over looping rounds (5 refinement loops).
3. Adjudication Agent: The 'ContextAwareJudge' (Judge, summary, score, feedback).

### Tool Integration & Data Sources:
#### Custom tools, wrap external services:
1. Google Scholar Tool(SerpApi) - academic search results.
2. PubMed Tool: Uses Biopython (Entrez/Medline) to query the NCBI Entrez biomedical database.

### Key Architectural Robustness:
1. Context Engineering: Implemented context compaction - pruning debater history, reduce latency and model costs.
2. State Management:
Uses an 'InMemorySessionService' to track and maintain ephemeral session state across different conversation stages.
3. Observability: Extensive logging is used across all function and tool calls to ensure full visibility into the execution flow.
4. Agent Evaluation: Includes a dedicated evaluation module 'agent_evaluation.py' using Gemini Similarity Scoring and fuzzy matching to verify the accuracy of tool-retrieved references.

```ThesisAdvisorAgent/<br>
├── app//
│  ├─ config//
│  │  └─ settings.py                # Configuration (Model versions, Logging)<br>
│  ├─ function_helpers//
│  │  └─ cloud_helpers.py
│  ├─ core//
│  │  ├─ agents.py                  # Definition of Tool-Use Agents (Scholar/PubMed)
│  │  └─ anylize_and_recommend.py   # The Debate & Judge Logic (PRO/CON/Judge)
│  └─ infrastructure//
│     └─ tools.py                   # Implementation of GoogleScholar and PubMed clients
├── thesis_advisor_client.py        # Main entry point (CLI application)
├── requirements.txt                # Dependencies
└── README.md/
```

### About deployment and usage
The deployed folder: "thesis_advisor_deploy"
Run with the Cloud Run: run demo_G_cloud_agent.py
Local Run: main.py
**.env file should have api_keys!**
**config as well the settings.py file**

deployed (via cmd, using venv, on root folder):
.\venv\Scripts\adk.exe deploy agent_engine --project=PROJECT_ID --region=us-central1 thesis_advisor_deploy --agent_engine_config_file=thesis_advisor_deploy/.agent_engine_config.json

## If I had more time - next steps & improvements
1. Currently, debaters instructions seems pretty good, but I would like to achieve even more sophisticated debate instructions, and implement a more accurate scoring method.
2. Evaluation was performed over the research tools' output, but not yet on the debate result itself.
That can be achieved by establish a reliable evaluation baseline by testing the debate's ability to independently justify a known, published thesis. This is achieved by concealing the definitive target article from the search tool results. The test aims to verify that the Pro Agent can still achieve a "PRO" verdict by successfully utilizing adjacent research and logical reasoning.
3. Clean Code (some last touch here on DRY principal, organize functions, and better exception handling)
4. Integrate more search tools.
5. Deploy the debater part to the cloud as well.
6. Optimization (token usage, while integrating more search tools, accurate Rate limiting using the API)
Better limiting of token usage for the debaters while allowing them to use more tools to search simultaneously.
7. Add GUI (Web front-end) users could insert API keys and then start using the program.
8. Add Metrics (Tokens, Speed).
9. Add Long memory persistent usage, e.g. add per user ID his thesis debates history for future reconsideration.
10. Consider using async instead to open the program to another functionality while waiting for debate result.

# Thank you for your guidance and instruction! That was an amazing course!

##### ~~~~~~~~
#### ~~~~~~~~
### ~~~~~~
## ~~~~~~
# ~~~~
# ~~~~
## ~~~~~~
### ~~~~~~
#### ~~~~~~~~
##### ~~~~~~~~

Irrelevant, kudos! :)
| | | | | | | | | | |
V V V V V V V V V V V
~~~ ~~~ ~~~ ~~~ ~~~ ~~~
## 📖 Overview
The Thesis Advisor AI is not just a chatbot; it is a structured workflow designed to mimic a rigorous academic consultation. It leverages Google Cloud Vertex AI and the Agent Development Kit (ADK) to perform a two-phase analysis:

Discovery Phase: A specialized "Talk Agent" searches for real academic literature (PubMed, Google Scholar) most relevant to your thesis.

Debate Phase: Two context-aware agents (PRO and CON) engage in a multi-round debate to analyze (e.g.) the feasibility, novelty, and ethics of your idea, supervised by a "Judge" agent that provides a final verdict and scoring.

## 🚀 Key Features
Hybrid Architecture: Combines deployed Vertex AI agents for tool execution with local, asynchronous coordination for complex debate logic.

Real Academic Grounding: Uses custom FunctionTools to query Google Scholar and PubMed, ensuring advice is based on existing literature, not hallucinations.

Multi-Agent Debate System:

Agent PRO: Argues for the feasibility and impact of the thesis.

Agent CON: Critiques methodology, scope, and ethical implications.

Context-Aware: Agents remember the opponent's previous points (of last round only, so it is fair, not but who answer last) but adhere to strict round-by-round rules (e.g., Opening, Rebuttal, Closing).

Automated Judgment: A neutral Judge agent reviews the entire debate transcript to assign scores and suggest actionable improvements.

Structured Output: Parses complex tool outputs into clean, readable citations and snippets.

## 🛠️ Architecture
The project is structured into Core logic and Infrastructure:

Plaintext


## 📦 Prerequisites
Before running the agent, ensure you have:

Python 3.10+ installed.

A Google Cloud Project with Vertex AI API enabled.

Google Cloud CLI installed and authenticated (gcloud auth application-default login).
 

## 🔧 Installation
Clone the Repository

```
git clone https://github.com/yourusername/thesis-advisor-agent.git
cd thesis-advisor-agent
```
Create a Virtual Environment

```python -m venv venv```
# Windows

```.\venv\Scripts\activate```
# Mac/Linux
```
source venv/bin/activate

Install Dependencies
```

```pip install -r requirements.txt``` (e.g. inside terminal)
Environment Setup The application relies on Google Cloud environment variables.<br> You can set them in your terminal or ensure thesis_advisor_client.py has the correct defaults:
Update setting file with your correct data,
You should have .env file in your root with your API-keys inserted:
```
GEMINI_API_KEY=
SERPAPI_API_KEY=
NCBI_CONTACT_EMAIL=
NCBI_API_KEY=
```

Deploy via cmd to Google cloud, run from root folder:
```
.\venv\Scripts\adk.exe deploy agent_engine --project="YOUR_PROJECT_ID" --region=us-central1 thesis_advisor_deploy --agent_engine_config_file=thesis_advisor_deploy/.agent_engine_config.json
```
(```us-central1``` is recommended, there are other options)
(To manage that, you should have google cloud profile, install 

export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
🏃 Usage
Run the main client application:


python thesis_advisor_client.py
The Workflow:
Input: Enter your thesis idea (e.g., "The impact of microplastics on gut microbiome diversity in coastal regions").

Search: The agent will use PubMed (for medical topics) or Google Scholar (for general topics) to find 5 relevant papers.

Review: You will see a formatted list of articles with snippets and links.

Action: You can choose to:

[Q]uit: Exit.

[R]efine: Rewrite your thesis and search again.

[C]ontinue: Launch the Debate


### The Debate Process (If you choose 'C' to continue'):
Criteria Selection: The agent asks you to select evaluation criteria (e.g., Feasibility, Novelty, Ethics). & are hard coded suggestions, one is for user if he would like to add new one.

### Debaters logic, trying to be fair and smart (the last debater don't have an automatic win due to say the last word:
```
Rules:
1. Initialization: Agents receive Thesis, Criteria, and Initial References.
2. Round 1 (Opening): Make claims pro\con thesis claims using ONLY Initial References according 3 or 4  criteria NO SEARCH TOOLS allowed.
3. Round 2 (Rebuttal): Refute opponent's R1. NO SEARCH.
4. Round 3 (Deepen): Refute R2 + Strengthen case.  SEARCH ALLOWED (Google Scholar/PubMed) to find new evidence.
5. Round 4 (Rebuttal): Refute R3.  NO SEARCH.
6. Round 5 (Closing): Ignore opponent. Final strong case.  SEARCH ALLOWED for final stats/facts.
7. Verdict: Judge Agent evaluates full transcript and would give concrete advice on how to improve your thesis.
```

🛡️ Security Note
This repository contains agent logic (agents.py) but excludes specific deployment configurations (.env or project id) to protect project credentials.

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
