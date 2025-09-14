
---

# KaLLaM – Motivational-Therapeutic Advisor

> **Note to future stupid self**: You will forget everything. This file exists so you don’t scream at your computer in six months. Read it first.

---

## 🚀 Quickstart

1. **Create venv**
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

2. **Upgrade pip**

   ```bash
   python -m pip install -U pip
   ```

3. **Install project (runtime only)**

   ```bash
   pip install -e .
   ```

4. **Install project + dev tools (pytest, mypy, ruff)**

   ```bash
   pip install -e .[dev]
   ```

5. **Run tests**

   ```bash
   pytest -q
   ```

---

## 📂 Project layout (don’t mess this up)

```
project-root/
├─ pyproject.toml         # dependencies & config (editable mode uses src/)
├─ README.md              # you are here
├─ src/
│  └─ kallam/
│     ├─ __init__.py
│     ├─ app/             # orchestrator wiring, chatbot manager
│     ├─ domain/          # agents, judges, orchestrator logic
│     └─ infra/           # db, llm clients, search, token counter
└─ tests/                 # pytest lives here
```

* `app/` = entrypoint, wires everything.
* `domain/` = core logic (agents, judges, orchestrator rules).
* `infra/` = all the boring adapters (DB, APIs, token counting).
* `tests/` = if you don’t write them, you’ll break everything and blame Python.

---

## 🔑 Environment variables

Put these in `.env` at project root:

```
OPENAI_API_KEY=sk-...
SEA_LION_API_KEY=...
AWS_ROLE_ARN=...
AWS_DEFAULT_REGION=ap-southeast-2
TAVILY_API_KEY=...
```

Load automatically via `python-dotenv`.

---

## 🧪 Common commands

* Run chatbot manager manually:

  ```bash
  python -m kallam.app.chatbot_manager
  ```

* Run lint:

  ```bash
  ruff check src tests
  ```

* Run type check:

  ```bash
  mypy src
  ```

* Export a session JSON (example):

  ```python
  from kallam.app.chatbot_manager import ChatbotManager
  mgr = ChatbotManager()
  sid = mgr.start_session()
  mgr.handle_message(sid, "hello world")
  mgr.export_session_json(sid)
  ```

---

## 🧹 Rules for survival

* Always activate `.venv` before coding.
* Never `pip install` globally, always `pip install -e .[dev]` inside venv.
* If imports fail → you forgot editable install. Run `pip install -e .` again.
* If SQLite locks up → delete `*.db` files and start fresh.
* If you break `pyproject.toml` → copy it from git history, don’t wing it.

---

## ☠️ Known pitfalls

* **Windows error “source not recognized”** → you’re not on Linux. Use `.venv\Scripts\Activate.ps1`.
* **“No module named kallam”** → you didn’t install with `-e`.
* **Tests can’t import your code** → run `pytest` from project root, not inside `tests/`.
* **Pip complains about `[dev]`** → you typo’d in `pyproject.toml`. Fix the `[project.optional-dependencies]` block.

---

That’s it. If you follow this, future you won’t rage-quit.

---

“The primary objective of this project is to demonstrate the effectiveness of SEALION models in handling psychological context and dialogue—both by generating meaningful therapeutic sessions with users and by evaluating conversations through automated SEALION-based MISC annotation as an AI coder.”