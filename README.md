
# KaLLaM – Motivational‑Therapeutic Advisor

KaLLaM is a bilingual (Thai/English) multi‑agent assistant for health and mental‑health conversations. It orchestrates specialized agents (Supervisor, Doctor, Psychologist, Translator, Summarizer), persists sessions in SQLite, and offers Gradio UIs plus evaluation utilities for MISC‑style conversation coding.


## Key Features

- Multi‑agent orchestration: Supervisor routes messages to domain experts.
- Bilingual support: Thai/English with SEA‑Lion powered translation.
- Persistence: SQLite session/message/summary stores with JSON export.
- UIs: Gradio demo and developer apps for quick experimentation.
- Evaluation: Scripts for SEA‑Lion based MISC silver coding (EN/TH).
- Pragmatic tooling: Logging, request tracing, token counting cache.


## Requirements

- Recommend Python 3.11+ as it might not work as we intended
- API access where used:
  - SEA‑Lion: `SEA_LION_API_KEY` (required for translation/summarization/flagging)
  - Google Gemini: `GEMINI_API_KEY` (required for Doctor/Psychologist agents)
  - Optional: OpenAI (`OPENAI_API_KEY`) if using OpenAI via `strands-agents`
  - Optional: AWS Bedrock (`AWS_DEFAULT_REGION`, `AWS_ROLE_ARN`) for `supervisor_bedrock.py`


## Installation

1) Create and activate a virtualenv

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Upgrade pip and install

```bash
python -m pip install -U pip
pip install -e .            # runtime
pip install -e .[dev]       # + pytest, ruff, mypy
```

3) Run tests

```bash
pytest -q
```


## Environment Variables

Create a `.env` at the project root. Load happens via `python-dotenv`.

```
# Core
SEA_LION_API_KEY=...
SEA_LION_BASE_URL=https://api.sea-lion.ai/v1   # optional (default)
SEA_LION_MODEL=aisingapore/Gemma-SEA-LION-v4-27B-IT   # optional for scripts
GEMINI_API_KEY=...

# Optional integrations
OPENAI_API_KEY=sk-...
AWS_DEFAULT_REGION=ap-southeast-2
AWS_ROLE_ARN=...
TAVILY_API_KEY=...  # if you wire search tooling
```


## Usage

Run a demo UI (recommended):

```bash
python gui/chatbot_demo.py
# or the Thai‑first developer app
python gui/chatbot_dev_app.py
```

Simple programmatic session workflow:

```python
from kallam.app.chatbot_manager import ChatbotManager

mgr = ChatbotManager()
sid = mgr.start_session(saved_memories="diabetes type‑2 | on metformin")

# … integrate your own routing or UI; see Gradio apps for examples …

# Export a session to JSON
path = mgr.export_session_json(sid)
print("Exported:", path)
```

## Project Layout

```
project-root/
├─ pyproject.toml            # deps and tool config
├─ src/kallam/
│  ├─ app/                   # ChatbotManager facade
│  ├─ domain/agents/         # supervisor, doctor, psychologist, translator, summarizer, orchestrator
│  └─ infra/                 # sqlite stores, exporter, token counter, config
├─ gui/                      # gradio apps (demo, dev, and simple example)
├─ scripts/                  # data tools: MISC coding, preprocessing, notebooks
├─ tests/                    # pytest and storage smoke UI
├─ exported_sessions/        # will be generated through export sessions as JSON
└─ Data/                     # datasets holder (in json jsonl and csv)
   ├─ orchestrated/
   ├─ human/
   └─ single-agent/
```


## Agents at a Glance

- Supervisor: routes, produces flags and final response scaffolding (SEA-Lion).
- Translator: SEA‑Lion backed Thai/English translation.
- Summarizer: SEA‑Lion backed conversation/health summaries.
- Doctor: Gemini backed medical guidance with safety guardrails.
- Psychologist: Thai via SEA‑Lion, English via Gemini; MI‑oriented.

Orchestrator configuration lives in `src/kallam/domain/agents/orchestrator.py` (models, language, thresholds).


## Data Persistence

- SQLite schema is created automatically in your chosen DB file (default `chatbot_data.db`).
- Stores: `SessionStore`, `MessageStore`, `SummaryStore` in `src/kallam/infra/`.
- Export: `JsonExporter` writes per‑session or all sessions to `exported_sessions/`.


## Development

- Lint: `ruff check src tests`
- Type check: `mypy src`
- Logs: written under `logs/` by each agent and manager; request tracing is enabled for key paths.
- Tips:
  - Use editable install (`pip install -e .[dev]`) to enable imports.
  - If SQLite locks up locally, stop processes using the DB or remove stale `*.db` files.
  - Run pytest from repo root so tests see `src/` via editable install.


## Evaluation and Scripts

- `scripts/eng_silver_misc_coder.py` and `scripts/thai_silver_misc_coder.py` implement SEA‑Lion based BiMISC/MISC silver coding with JSON‑only enforcement and metrics. See file headers for dataset and output formats.
- `scripts/model_evaluator.py`, preprocessing utilities, and `scripts/visualizer.ipynb` support dataset prep and analysis.


## Citation

References and datasets used by this project are listed in `Citation.md`.


## License

Apache License 2.0. See `LICENSE` for details.
